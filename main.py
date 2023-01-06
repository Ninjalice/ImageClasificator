# Luego, importamos las librerías necesarias
import tensorflow as tf
from tensorflow import keras

# Descargamos el dataset MNIST Fashion
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Preprocesamos los datos
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Definimos la estructura de la red neuronal
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# Compilamos la red
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenamos la red
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

results = model.evaluate(x_test,  y_test, verbose = 0)
print('test loss, test acc:', results)