import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D


train_dir = r"C:\Users\NavasChalattil\Desktop\facemask\dataset\train"
val_dir   = r"C:\Users\NavasChalattil\Desktop\facemask\dataset\val"
test_dir  = r"C:\Users\NavasChalattil\Desktop\facemask\dataset\test"


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(224,224), batch_size=32, class_mode="binary"
)

val_gen = val_test_datagen.flow_from_directory(
    val_dir, target_size=(224,224), batch_size=32, class_mode="binary"
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir, target_size=(224,224), batch_size=32, class_mode="binary", shuffle=False
)


base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
preds = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=preds)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])


print("🔹 Training Phase 1: Training with frozen base model...")
history = model.fit(
    train_gen, validation_data=val_gen,
    epochs=10
)


base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False  

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy",
              metrics=["accuracy"])

print("🔹 Training Phase 2: Fine-tuning top layers...")
history_ft = model.fit(
    train_gen, validation_data=val_gen,
    epochs=10
)


loss, acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {acc*100:.2f}%")


model.save("mask_detector_strong.h5")
print("🎉 Model saved as mask_detector_strong.h5")
