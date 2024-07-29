import keras
from keras import layers
from tensorflow import data as tf_data
import tensorflow as tf

# Configuration GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12288)])
    except RuntimeError as e:
        print(e)

image_size = (180, 180)
batch_size = 16  # Réduire la taille du batch pour économiser la mémoire

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "/mnt/c/Users/monta/Desktop/BODMAS2/Graphes_entropie",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale'  # Assurer que les images sont chargées en mode niveaux de gris
)

# Préchargement réduit
train_ds = train_ds.prefetch(1)
val_ds = val_ds.prefetch(1)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # Bloc d'entrée
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)  # Utiliser softmax pour la classification multi-classes

    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (1,), num_classes=14)  # Ajuster input_shape pour les images en niveaux de gris

# Tracer le modèle
#keras.utils.plot_model(model, show_shapes=True)

epochs = 5

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]

model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # Utiliser SparseCategoricalCrossentropy
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

# Sauvegarder le modèle entraîné dans un fichier
model.save('entropie_model.h5')
