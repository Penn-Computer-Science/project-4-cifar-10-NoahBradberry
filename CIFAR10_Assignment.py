import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
# Print version to verify tf is installed
print(tf.__version__)

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


#Check to make sure there are no values that are NAN (Not a Number)

print("Any NaN Training:", np.isnan(x_train).any())
print("Any NaN Testing:", np.isnan(x_test).any())

#tell the model what shape to expect
input_shape = (32, 32, 3) #28x28 pixels, 1 color channel (grayscale)

#Reshape the Data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


#Convert our labels to be ONE-HOT, not sparse
y_train = y_train.flatten()
y_test = y_test.flatten()

#Show an example image from Mnist
#plt.imshow(x_train[random.randint(0, 59999)][:,:,0], cmap= "gray")
#plt.show()

batch_size = 64
num_classes = 10
epochs = 10

#Build the Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train,epochs=10,validation_data=(x_test, y_test))

#Plot out traing and validation accuracy and loss
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], 'b', label = 'Training Loss')
ax[0].plot(history.history['val_loss'], 'r', label = 'Validation Loss')
legend = ax[0].legend(loc = 'best', shadow = True)
ax[0].set_title("Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")

ax[1].plot(history.history['acc'], color = 'b', label = 'Training Acuracy')
ax[1].plot(history.history['val_acc'], color = 'r', label = 'Validation Acuracy')
legend = ax[1].legend(loc = 'best', shadow = True)
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Acuracy")

plt.tight_layout()
plt.show()