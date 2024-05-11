import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def MLP():
    inputs = Input(shape=(28 * 28,))
    dense1 = Dense(512, activation='relu')(inputs)
    dropout1 = Dropout(0.2)(dense1)
    dense2 = Dense(256, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    outputs = Dense(10, activation='softmax')(dropout2)
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = MLP()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, 28 * 28)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28 * 28)).astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()