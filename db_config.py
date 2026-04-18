import mysql.connector

def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="faizun354",
        database="platycerium_detection"
    )
    return conn