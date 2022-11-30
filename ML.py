import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataset = np.loadtxt('pulsar_data.csv', delimiter=',', skiprows=1)

X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()
model.add(Dense(1, input_shape=(8,), activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
history = model.fit(X, Y, validation_split=0.2, epochs=100, batch_size=20)

_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))

plt.plot(history.history['accuracy'])
plt.show()