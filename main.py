import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Dropout 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
import numpy as np 
from tensorflow.keras.preprocessing import image 


img_size = 224
batch_size = 32 

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    "dataset",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed =42
)

val_data = datagen.flow_from_directory(
    "dataset",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=42
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("Train samples:", train_data.samples)
print("Validation samples:", val_data.samples)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

model.save('Image_Classification.h5')

# Predictions
img=image.load_img('Figure_1.png',target_size=(img_size,img_size))
img_array=img.img_to_array(img)/255.0
img_array=np.expand_dims(img_array,axis=0)

predictions=model.predict(img_array)
class_names=list(train_data.class_indices.keys())

print('Predicted Class:',class_names[np.argmax(predictions)])
