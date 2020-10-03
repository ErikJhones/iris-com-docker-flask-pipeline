from flask import Flask, request, jsonify
from os import path
from decouple import config as cfg #para variavel de ambiente
import argparse, joblib
from utils import load_models, check_inputs
from train import load_data
import numpy as np

app = Flask(__name__)

#load env vars (ao usar isso, comentar o if main pra executar com flask run)
#HOST = cfg('HOST', cast=str)
#PORT = cfg('PORT', cast=str)
#DEBUG = cfg('DEBUG', cast=bool)

#load models
model, tf = load_models()

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        #X,y = load_data()
        #x = X[1,:].reshape(1, -1)
        #print(np.array(request.json["features"]).reshape(1-1))
        x = check_inputs(request.json['features'])
        
        #x = np.array(request.json['features']).reshape(1,-1)
        y_hat = model.predict(tf.transform(x))

        return jsonify(output={"y_hat": y_hat.tolist()}, status=200, message="Model Working")

@app.route('/predict_test')
def predict_test():
    X,y = load_data()
    X_tf = tf.transform(X)
    y_hat = model.predict(X_tf)
    return 'Predict'

@app.route('/')
def hello_world():
    return 'Hello, Carol'


#if __name__ == "__main__":
 #   parser = argparse.ArgumentParser(description='Iris classifier api 0.0.1')
  #  parser.add_argument('--host', default='localhost', type=str)
   # parser.add_argument('--port', default=5000, type=str)
    #parser.add_argument('--debug', default=True, type=str)
    #args = vars(parser.parse_args())

    #load vars
    #model, tf = load_models()

#app.run(host=args['host'], port=args['port'], debug=args['debug'])


    #https://simpleisbetterthancomplex.com/2015/11/26/package-of-the-week-python-decouple.html
    #https://flask.palletsprojects.com/en/1.1.x/quickstart/#quickstart
    #https://scikit-learn.org/stable/modules/compose.html#combining-estimators