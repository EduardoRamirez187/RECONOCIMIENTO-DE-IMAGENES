import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Configuración de parámetros
ancho_imagen, alto_imagen = 128, 128  
tamano_lote = 32  
epocas = 50

# Carpetas donde se encuentran las imágenes
directorio_entrenamiento = "C:/Users/Eduardo/Desktop/Archivo/dataset/entrenamiento" 
directorio_validacion = "C:/Users/Eduardo/Desktop/Archivo/dataset/validacion" 

# Generador de imágenes con aumentación
generador_entrenamiento = ImageDataGenerator(
    rescale = 1.0 / 255,
    rotation_range = 50,
    width_shift_range = 0.3,
    height_shift_range = 0.3,
    shear_range = 15,
    zoom_range = [0.5, 1.5],
    vertical_flip = True,
    horizontal_flip = True,
)

generador_validacion = ImageDataGenerator(rescale=1.0 / 255)

# Carga de datos
datos_entrenamiento = generador_entrenamiento.flow_from_directory(
    directorio_entrenamiento,
    target_size=(ancho_imagen, alto_imagen),
    batch_size=tamano_lote,
    class_mode="categorical",  # Cambio a 'categorical'
)

datos_validacion = generador_validacion.flow_from_directory(
    directorio_validacion,
    target_size=(ancho_imagen, alto_imagen),
    batch_size=tamano_lote,
    class_mode="categorical",  # Cambio a 'categorical'
)

# Creación del modelo de red neuronal convolucional
RED_NEURONAL_CONVOLUCIONAL = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(ancho_imagen, alto_imagen, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),
    
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    
    # Nueva capa de salida para 4 clases
    layers.Dense(5, activation="softmax"),  # Cambio: 4 neuronas y activación softmax
])

# Compilación del modelo
RED_NEURONAL_CONVOLUCIONAL.compile(
    optimizer="adam",
    loss="categorical_crossentropy",  # Cambio: categorical_crossentropy
    metrics=["accuracy"]
)

# Entrenamiento del modelo
entrenar_red = RED_NEURONAL_CONVOLUCIONAL.fit(
    datos_entrenamiento,
    epochs=epocas,
    validation_data=datos_validacion,
)

# Evaluación y guardado del modelo
puntaje = RED_NEURONAL_CONVOLUCIONAL.evaluate(datos_validacion)
print(f"Pérdida: {puntaje[0]:.4f}, Precisión: {puntaje[1]:.4f}")

RED_NEURONAL_CONVOLUCIONAL.save("modelo_clasificador_frutas.h5")
RED_NEURONAL_CONVOLUCIONAL.save_weights("pesos_frutas.weights.h5")
