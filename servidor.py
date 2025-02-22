from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as img_preprocessing

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model("modelo_clasificador_frutas.h5")
clases = ["cereza", "mango", "manzana", "naranja", "platano"]

# Inicializar Flask
servidor = Flask(__name__)

# Ruta principal
@servidor.route("/")
def index():
    return render_template("index.html")

# Función para capturar la cámara y hacer detección
def generar_frames():
    cap = cv2.VideoCapture(0)  # Activar la cámara

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocesar la imagen para el modelo
        frame_resized = cv2.resize(frame, (128, 128))
        img_array = img_preprocessing.img_to_array(frame_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Hacer predicción
        prediccion = modelo.predict(img_array)
        probabilidad_maxima = np.max(prediccion)
        clase_predicha = clases[np.argmax(prediccion)]

        # Dibujar resultado si la probabilidad es alta
        if probabilidad_maxima > 0.9:
            cv2.putText(frame, f"{clase_predicha}: {probabilidad_maxima*100:.2f}%", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (50, 50), (400, 400), (0, 255, 0), 2)

        # Convertir el frame a JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        # Enviar el frame en formato de transmisión
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Ruta para la detección en tiempo real
@servidor.route("/deteccion")
def deteccion():
    return Response(generar_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    servidor.run(port=4000, debug=True)
