from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import numpy
seed=7
numpy.random.seed(seed)
dataset=numpy.loadtxt("../data/pima-indians-diabetes.data.csv",delimiter=",")
x_train=dataset[0:450,0:8]
y_train=dataset[0:450,8]

#Create Model
model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#Compile
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy','accuracy'])

#Fit Model
hist=model.fit(x_train,y_train,epochs=120,batch_size=10)

#plot metrics
pyplot.plot(hist.history['acc'])
pyplot.show()

#evaluate the model
x_eval=dataset[451:,0:8]
y_eval=dataset[451:,8]
scores=model.evaluate(x_eval,y_eval)
print("\n%s : %.2f%%" % (model.metrics_names[1],scores[1]*100))
print("\n%s : %.2f%%" % (model.metrics_names[2],scores[2]*100))

#Serialize the models
model_json=model.to_json()
with open("../serialize/model.json","w") as json_file:
	json_file.write(model_json)
#Serialize weights to HDF5
model.save_weights("../serialize/model.h5")
print("Weights Saved")


