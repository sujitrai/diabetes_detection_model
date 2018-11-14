import os
import sys
import logging
from flask import Flask, request, json

from serv import get_model_api

#define app
app = Flask(__name__)

model_api=get_model_api()

#API route
@app.route('/')
def index():
	return "INDEX API"


@app.route('/prdict',methods=['POST'])
def predict():
	app.logger.info(request.json)
	input=request.json
	output=model_api(input)
	op=json.dumps(output)
	return op

@app.errorhandler(404)
def url_error(e):
	return """
	wrong url!! """.format(e),404

@app.errorhandler(500)
def server_error(e):
	return """ internal error occurred""".format(e),500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=False) 
 
