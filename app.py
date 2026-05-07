from flask import Flask, render_template, request, redirect, session, url_for, send_file
from flask import flash
import mysql.connector

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

import numpy as np
import cv2
import os
import tensorflow as tf
import io
import uuid

from datetime import datetime, timedelta

from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash

from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage

# ======================
# APP CONFIG
# ======================
app = Flask(__name__)

app.secret_key = os.environ.get("SECRET_KEY", "skripsi_platycerium")

# AUTO CREATE FOLDER
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/gradcam", exist_ok=True)

# ======================
# EMAIL CONFIG
# ======================
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True

app.config['MAIL_USERNAME'] = os.environ.get("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.environ.get("MAIL_PASSWORD")
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get("MAIL_USERNAME")

mail = Mail(app)

# ======================
# MODEL
# ======================
print("LOAD MODEL...")
model = load_model('model/model_platycerium_bismillah.keras')
print("MODEL BERHASIL")

# ======================
# DATABASE
# ======================
def get_db_connection():
    return mysql.connector.connect(
        host=os.environ.get("crossover.proxy.rlwy.net"),
        user=os.environ.get("root"),
        password=os.environ.get("FOyBTXaXIeRUcIhMJxyclhMMyYTNJToU"),
        database=os.environ.get("railway"),
        port=int(os.environ.get("25539", 3306))
    )

# ======================
# LOGIN
# ======================
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():

    username = request.form['username']
    password = request.form['password']

    conn = get_db_connection()
    cursor = conn.cursor(buffered=True)

    cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
    user = cursor.fetchone()

    if user and check_password_hash(user[2], password):
        session['login'] = True
        return redirect('/dashboard')
    else:
        return redirect('/')

# ======================
# FORGOT PASSWORD
# ======================
@app.route('/forgot_success')
def forgot_success():
    return render_template('forgot_success.html')

@app.route('/forgot')
def forgot():
    return render_template('forgot.html')

@app.route('/forgot', methods=['POST'])
def forgot_post():

    email = request.form['email']

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cursor.fetchone()

    if not user:
        cursor.close()
        conn.close()

        flash(
            "Email tidak terdaftar! hubungi admin.",
            "danger"
        )

        return redirect('/forgot')

    token = str(uuid.uuid4())
    expired = datetime.now() + timedelta(minutes=15)

    cursor.execute("""
    UPDATE users
    SET reset_token=%s, token_expired=%s
    WHERE email=%s
    """, (token, expired, email))

    conn.commit()

    reset_link = request.host_url + f"reset/{token}"

    msg = Message(
        subject="Reset Password PlatyScan AI",
        recipients=[email]
    )

    msg.body = f"""
Halo,

Klik link berikut untuk reset password:
{reset_link}

Link berlaku 15 menit.
"""

    mail.send(msg)

    cursor.close()
    conn.close()

    return redirect('/forgot_success')

# ======================
# RESET PASSWORD
# ======================
@app.route('/reset/<token>')
def reset(token):

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
    SELECT * FROM users
    WHERE reset_token=%s AND token_expired > NOW()
    """, (token,))

    user = cursor.fetchone()

    if not user:
        return "Token tidak valid / expired"

    return render_template("reset.html", token=token)

@app.route('/reset', methods=['POST'])
def reset_post():

    token = request.form['token']
    password = request.form['password']

    hashed_password = generate_password_hash(password)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE users
    SET password=%s,
        reset_token=NULL,
        token_expired=NULL
    WHERE reset_token=%s
    """, (hashed_password, token))

    conn.commit()

    cursor.close()
    conn.close()

    return redirect('/')

# ======================
# DASHBOARD
# ======================
@app.route('/dashboard')
def dashboard():

    if "login" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM history")
    total_prediksi = cursor.fetchone()[0]

    cursor.execute("""
    SELECT COUNT(*)
    FROM history
    WHERE diagnosis != 'sehat'
    """)

    total_tidak_sehat = cursor.fetchone()[0]

    cursor.execute("""
    SELECT AVG(confidence)
    FROM history
    """)

    avg_confidence = cursor.fetchone()[0]

    if avg_confidence is None:
        avg_confidence = 0

    cursor.execute("""
    SELECT diagnosis, COUNT(*)
    FROM history
    GROUP BY diagnosis
    """)

    chart = cursor.fetchall()

    labels = [row[0] for row in chart]
    values = [row[1] for row in chart]

    cursor.close()
    conn.close()

    return render_template(
        "dashboard.html",
        total_prediksi=total_prediksi,
        total_tidak_sehat=total_tidak_sehat,
        avg_confidence=avg_confidence,
        labels=labels,
        values=values
    )

# ======================
# PREDIKSI PAGE
# ======================
@app.route('/prediksi')
def prediksi():

    if "login" not in session:
        return redirect(url_for("login"))

    return render_template('prediksi.html')

# ======================
# PREDICT
# ======================
@app.route('/predict', methods=['POST'])
def predict():

    if "login" not in session:
        return redirect(url_for("login"))

    file = request.files['gambar']
    id_tanaman = request.form['id_tanaman']

    filename = str(uuid.uuid4()) + "_" + file.filename

    filepath = os.path.join("static/uploads", filename)
    file.save(filepath)

    db_image_path = "uploads/" + filename

    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))

    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    confidence = float(np.max(prediction))
    class_index = np.argmax(prediction[0])

    classes = [
        "bercak_coklat",
        "bercak_putih",
        "busuk_akar",
        "sehat",
        "sunburn"
    ]

    diagnosis = classes[class_index]

    penanganan_dict = {
        "bercak_coklat": "Tingkatkan sirkulasi udara.",
        "bercak_putih": "Gunakan alkohol isopropil 70%.",
        "busuk_akar": "Hentikan penyiraman segera.",
        "sehat": "Tanaman dalam kondisi sehat.",
        "sunburn": "Pindahkan dari sinar matahari langsung."
    }

    penanganan = penanganan_dict[diagnosis]

    # ======================
    # GRADCAM
    # ======================
    heatmap = make_gradcam_heatmap(img_array, model, "conv_1")

    gradcam_path = os.path.join("static/gradcam", filename)
    db_gradcam_path = "gradcam/" + filename

    save_gradcam(filepath, heatmap, gradcam_path)

    # ======================
    # SAVE DATABASE
    # ======================
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO history
    (id_tanaman, diagnosis, confidence,
    image_input, gradcam_image, penanganan)

    VALUES (%s,%s,%s,%s,%s,%s)
    """, (
        id_tanaman,
        diagnosis,
        confidence,
        db_image_path,
        db_gradcam_path,
        penanganan
    ))

    conn.commit()

    cursor.close()
    conn.close()

    return render_template(
        "prediksi.html",
        hasil=diagnosis,
        confidence=confidence,
        penanganan=penanganan,
        gambar=filename,
        gradcam=filename
    )

# ======================
# GRADCAM FUNCTION
# ======================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        predictions = tf.convert_to_tensor(predictions)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.math.reduce_max(heatmap)

    if max_val == 0:
        return heatmap.numpy()

    heatmap /= max_val

    return heatmap.numpy()

def save_gradcam(img_path, heatmap, cam_path):

    img = cv2.imread(img_path)

    heatmap = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed_img = cv2.addWeighted(
        img,
        0.6,
        heatmap,
        0.4,
        0
    )

    cv2.imwrite(cam_path, superimposed_img)

# ======================
# LOGOUT
# ======================
@app.route('/logout')
def logout():

    session.clear()

    return redirect(url_for("login"))

# ======================
# RUN APP
# ======================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
