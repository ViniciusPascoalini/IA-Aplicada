# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


# Create callbacks
#callbacks = [ModelCheckpoint(('model.h5'), save_best_only=True, 
#                             save_weights_only=False)]
             
# mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
# (x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

# x_train = x_train/255.0
# x_test = x_test/255.0

print(x_train.shape)
print(x_train[0].shape)

model = Sequential()
model.add(LSTM(200, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(200, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(16, activation='softmax'))


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(x_train,
          y_train,
          epochs=50,
          #callbacks=callbacks,
          validation_data=(x_test, y_test))

# model.fit(x_train[0:400],
#           y_train[0:400],
#           epochs=1000,
#           # callbacks=callbacks,
#           validation_data=(x_test[450:550], y_test[450:550]))
