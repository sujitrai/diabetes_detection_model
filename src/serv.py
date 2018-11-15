from keras.models import Sequential
from keras.models import model_from_json
from numpy import array
def get_model_api():
	""" """
	json_file=open("../serialize/model.json","r")
	model_json=json_file.read()
	json_file.close()
	model=model_from_json(model_json)
	model.load_weights("../serialize/model.h5");
	print("Model has been loaded from file")
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	

	def model_api(input_data):
		""" """
		result={}
		result['input']=input_data
		id=array(input_data)
		preds=model.predict(id)
		rounded=[round(item[0]) for item in preds]
		result['output']=rounded.tolist()
		return result

	return model_api

