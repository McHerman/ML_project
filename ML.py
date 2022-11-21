from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# load the dataset
dataset = loadtxt('pulsar_data_training.csv', delimiter=',',skiprows=1)
# split into input (X) and output (y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

validation = loadtxt('pulsar_data_test.csv', delimiter=',',skiprows=1)

vX = validation[:,0:8]
vY = validation[:,8]

# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
history = model.fit(X, Y, validation_split = 0.33, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(vX, vY)
print('Accuracy: %.2f' % (accuracy*100))
print(history)