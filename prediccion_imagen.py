import tensorflow as tf
import numpy as np
import cv2

# Cargar el modelo previamente guardado
modelo = tf.keras.models.load_model("modelo_clasificador_mangos_naranjas.h5")

# Mapeo de índices a nombres de clases
clases = ["mango", "naranja"]

# Iniciar la cámara web
cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame.")
        break

    # Redimensionar el frame al tamaño esperado por la red (128x128)
    frame_redimensionado = cv2.resize(frame, (128, 128))

    # Convertir el frame de BGR a RGB
    frame_rgb = cv2.cvtColor(frame_redimensionado, cv2.COLOR_BGR2RGB)

    # Convertir el frame a un array de numpy
    img_array = np.array(frame_rgb)

    # Expandir dimensiones para que coincida con el formato esperado por el modelo (batch_size, alto, ancho, canales)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalizar los valores de los píxeles (como se hizo en el entrenamiento)
    img_array = img_array / 255.0

    # Hacer la predicción
    prediccion = modelo.predict(img_array)

    # Obtener la clase con mayor probabilidad y el porcentaje de confianza
    indice_predicho = np.argmax(prediccion[0])  # Índice de la clase con mayor probabilidad
    confianza = prediccion[0][indice_predicho] * 100  # Porcentaje de confianza

    # Mostrar el resultado en el frame
    texto_prediccion = f"Es un(a) {clases[indice_predicho]} ({confianza:.2f}%)"
    cv2.putText(frame, texto_prediccion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Encerrar el objeto detectado en un rectángulo si la confianza es mayor a 18%
    if confianza > 18:
        # Obtener las dimensiones del frame original
        alto, ancho, _ = frame.shape

        # Dibujar un rectángulo alrededor del objeto detectado
        cv2.rectangle(frame, (0, 0), (ancho, alto), (0, 255, 0), 2)  # Rectángulo verde

    # Mostrar el frame en una ventana
    cv2.imshow("Clasificación en tiempo real", frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()