import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- DATA PREPROCESSING ---
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    'dataset',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val = datagen.flow_from_directory(
    'dataset',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# --- MODEL ---
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(224,224,3)),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(5,activation='softmax')
])

# --- COMPILE ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- TRAIN (FAST) ---
model.fit(
    train,
    validation_data=val,
    epochs=1   # 🔥 FAST TRAINING
)

# --- SAVE MODEL ---
model.save("model.h5")

print("✅ Training Done & Model Saved!")