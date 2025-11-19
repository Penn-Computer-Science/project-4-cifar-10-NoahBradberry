import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()


#Check to make sure there are no values that are NAN (Not a Number)

print("Any NaN Training:", np.isnan(x_train).any())
print("Any NaN Testing:", np.isnan(x_test).any())

#tell the model what shape to expect
input_shape = (32, 32, 3) #28x28 pixels, 1 color channel (grayscale)

#Reshape the Data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


#Convert our labels to be ONE-HOT, not sparse
y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)


batch_size = 64
num_classes = 100
epochs = 10

#Build the Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train,epochs=epochs,validation_data=(x_test, y_test), validation_split = 0.1)

#Plot out traing and validation accuracy and loss
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], 'b', label = 'Training Loss')
ax[0].plot(history.history['val_loss'], 'r', label = 'Validation Loss')
legend = ax[0].legend(loc = 'best', shadow = True)
ax[0].set_title("Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")

ax[1].plot(history.history['accuracy'], color = 'b', label = 'Training Acuracy')
ax[1].plot(history.history['val_accuracy'], color = 'r', label = 'Validation Acuracy')
legend = ax[1].legend(loc = 'best', shadow = True)
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Acuracy")

plt.tight_layout()
plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

#generate the confusion matrix
# Predict the values from the testing dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis=1) 
# Convert testing observations to one hot vectors
Y_true = np.argmax(y_test, axis=1)
# compute the confusion matrix
confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes) 

# Define class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(100, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
