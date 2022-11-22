import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
#from numpy import loadtxt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# load the dataset
dataset = np.loadtxt('pulsar_data.csv', delimiter=',',skiprows=1)
# split into input (X) and outpu½t (y) variables
X = dataset[:13423,0:8]
Y = dataset[:13423,8]

vX = dataset[13424:,0:8]
vY = dataset[13424:,8] 


# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
history = model.fit(X, Y, epochs=20, batch_size=100)
weights = model.get_weights()
# evaluate the keras model
_, accuracy = model.evaluate(vX, vY)
print('Accuracy: %.2f' % (accuracy*100))
#print(weights)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.plot(weights)

plt.show()