import matplotlib.pyplot as plt
import tensorflow as tf
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(x_train.shape)
#
# plt.imshow(x_train[0], cmap="gray")
# plt.show()

# reshape the images from 2d to 3d for training
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# split the training data to training and validation splits using sklearn train_test_split
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                      test_size=0.05, shuffle=True, random_state=42)
#
# print('X_train.shape =>', X_train.shape)
# print('X_valid.shape =>', X_valid.shape)
# print('y_train.shape =>', y_train.shape)
# print('y_valid.shape =>', y_valid.shape)

# apply image transformations (augment the dataset) to make model robust to changes
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    validation_split=0)

# model = tf.keras.Sequential([
#   tf.keras.layers.Rescaling(1./255, input_shape=[28, 28, 1]),
#   tf.keras.layers.Conv2D(16, 3, padding='valid', activation='relu'),
#   tf.keras.layers.MaxPooling2D(2),
#   tf.keras.layers.Conv2D(32, 3, padding='valid', activation='relu'),
#   tf.keras.layers.MaxPooling2D(2),
#   tf.keras.layers.Conv2D(64, 3, padding='valid', activation='relu'),
#   tf.keras.layers.MaxPooling2D(2),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.5),
#   tf.keras.layers.Dense(64, activation='relu'),
#   tf.keras.layers.Dropout(0.5),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_valid, y_valid),
#     epochs=1)
#
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# batch_size = 128
# num_classes = 10
# epochs = 10
#
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam' , metrics=['accuracy'])
# hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
# print("The model has successfully trained")
#
# model.save('mnist.h5')
# print("Saving the model as mnist.h5")

model_ann = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=[28, 28, 1]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
print(model_ann.summary())
#
#
model_ann.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model_ann.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=2)
score = model_ann.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#
model_ann.save('mnist_ann.h5')
print("Saving the model as mnist_ann.h5")
