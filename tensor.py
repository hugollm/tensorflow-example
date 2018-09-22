import numpy as np
import tensorflow as tf


model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

model.compile(
    optimizer=tf.train.RMSPropOptimizer(0.001),
    loss='mse',
    metrics=['mae'],
)

train_input = np.array([
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
])
train_target = np.array([6, 7, 8])
test_input = np.array([
    [4, 5, 6, 7, 8],
    [5, 6, 7, 8, 9],
    [6, 7, 8, 9, 10],
])

model.fit(train_input, train_target, epochs=500, validation_split=0.2, verbose=0)

predicted = model.predict(test_input)

print('wanted: {}, got: {}'.format('09', predicted[0][0]))
print('wanted: {}, got: {}'.format('10', predicted[1][0]))
print('wanted: {}, got: {}'.format('11', predicted[2][0]))
