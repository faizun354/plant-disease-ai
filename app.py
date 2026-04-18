from flask import Flask, render_template, request, redirect, session, url_for
import mysql.connector

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import cv2
import os

import tensorflow as tf
import matplotlib.cm as cm
import pandas as pd
from flask import send_file
import io

model = load_model('model/model_platycerium_final.keras')
app = Flask(__name__)
app.secret_key = "skripsi_platycerium"

def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="faizun354",
        database="platycerium_detection"
    )
    return conn

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():

    username = request.form['username']
    password = request.form['password']

    conn = get_db_connection()
    cursor = conn.cursor(buffered=True)

    query = "SELECT * FROM users WHERE username=%s AND password=%s"

    cursor.execute(query,(username,password))
    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user:
        session['login'] = True
        return redirect('/dashboard')
    else:
        return redirect('/')
    
@app.route('/dashboard')
def dashboard():

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="faizun354",
        database="platycerium_detection"
    )

    cursor = conn.cursor()

    # total prediksi
    cursor.execute("SELECT COUNT(*) FROM history")
    total_prediksi = cursor.fetchone()[0]

    # data distribusi diagnosis untuk bar chart
    cursor.execute("""
    SELECT diagnosis, COUNT(*)
    FROM history
    GROUP BY diagnosis
    """)

    chart = cursor.fetchall()

    labels = [row[0] for row in chart]
    values = [row[1] for row in chart]

    conn.close()

    if "login" not in session:
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html",
        total_prediksi=total_prediksi,
        labels=labels,
        values=values
    )

@app.route('/prediksi')
def prediksi():
    if "login" not in session:
        return redirect(url_for("login"))
    return render_template('prediksi.html')

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['gambar']
    id_tanaman = request.form['id_tanaman']

    filename = file.filename

    filepath = os.path.join("static/uploads", filename)
    file.save(filepath)

    db_image_path = "uploads/" + filename

    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    confidence = float(np.max(prediction))
    class_index = np.argmax(prediction[0])

    classes = ["bercak_coklat","bercak_putih","busuk_akar","sehat","sunburn"]

    diagnosis = classes[class_index]
    penanganan_dict = {

    "bercak_coklat":
    "Potong daun yang terinfeksi dan semprot fungisida berbahan aktif mankozeb.",

    "bercak_putih":
    "Bersihkan area daun dan semprot fungisida ringan untuk mencegah penyebaran.",

    "busuk_akar":
    "Ganti media tanam dan kurangi penyiraman agar akar tidak terlalu lembab.",

    "sehat":
    "Tanaman dalam kondisi sehat. Lanjutkan perawatan rutin.",

    "sunburn":
    "Pindahkan tanaman ke tempat teduh dan kurangi paparan sinar matahari langsung."
    }

    penanganan = penanganan_dict[diagnosis]

    # =====================
    # GRADCAM
    # =====================

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        "conv_1"   # layer terakhir mobilenetv3
    )
    print("Heatmap min:", heatmap.min())
    print("Heatmap max:", heatmap.max())

    gradcam_path = os.path.join("static/gradcam", filename)
    db_gradcam_path = "gradcam/" + filename

    save_gradcam(filepath, heatmap, gradcam_path)
    
    conn = get_db_connection()
    cursor = conn.cursor(buffered=True)

    query = """
    INSERT INTO history
    (id_tanaman, diagnosis, confidence, image_input, gradcam_image)
    VALUES (%s,%s,%s,%s,%s)
    """

    cursor.execute(query, (
        id_tanaman,
        diagnosis,
        confidence,
        db_image_path,
        db_gradcam_path
    ))

    conn.commit()
    cursor.close()
    conn.close()

    if "login" not in session:
        return redirect(url_for("login"))

    return render_template(
        "prediksi.html",
        hasil=diagnosis,
        confidence=confidence,
        penanganan=penanganan,
        gambar=filename,
        gradcam=filename
    )

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap,0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def save_gradcam(img_path, heatmap, cam_path):

    img = cv2.imread(img_path)

    # resize heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # convert heatmap ke 0-255
    heatmap = np.uint8(255 * heatmap)

    # apply colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # overlay heatmap dengan gambar
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(cam_path, superimposed_img)

@app.route('/history')
def history():

    keyword = request.args.get('search')

    conn = get_db_connection()
    cursor = conn.cursor(buffered=True)

    if keyword:
        cursor.execute("""
        SELECT id_history, waktu, id_tanaman, diagnosis, confidence, image_input, gradcam_image
        FROM history
        WHERE id_tanaman LIKE %s
        ORDER BY waktu DESC
        """, ("%" + keyword + "%",))
    else:
        cursor.execute("""
        SELECT id_history, waktu, id_tanaman, diagnosis, confidence, image_input, gradcam_image
        FROM history
        ORDER BY waktu DESC
        """)

    data = cursor.fetchall()

    cursor.close()
    conn.close()
    if "login" not in session:
        return redirect(url_for("login"))

    return render_template(
        "history.html",
        data=data
    )

@app.route('/download_excel')
def download_excel():

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="faizun354",
        database="platycerium_detection"
    )

    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM history")
    data = cursor.fetchall()

    df = pd.DataFrame(data)

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='History')

    output.seek(0)
    if "login" not in session:
        return redirect(url_for("login"))

    return send_file(
        output,
        download_name="riwayat_prediksi.xlsx",
        as_attachment=True
    )

@app.route('/delete_history/<int:id>')
def delete_history(id):

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="faizun354",
        database="platycerium_detection"
    )

    cursor = conn.cursor()

    cursor.execute("DELETE FROM history WHERE id_history = %s", (id,))
    conn.commit()

    cursor.close()
    conn.close()
    if "login" not in session:
        return redirect(url_for("login"))

    return redirect('/history')

@app.route('/delete_all')
def delete_all():

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="faizun354",
        database="platycerium_detection"
    )

    cursor = conn.cursor()

    cursor.execute("DELETE FROM history")
    conn.commit()

    cursor.close()
    conn.close()
    if "login" not in session:
        return redirect(url_for("login"))

    return redirect('/history')

@app.route('/help')
def help():
    if "login" not in session:
        return redirect(url_for("login"))
    return render_template("help.html")


@app.route('/jurnal')
def jurnal():
    if "login" not in session:
        return redirect(url_for("login"))
    return render_template("jurnal.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == '__main__':
    app.run(debug=True)