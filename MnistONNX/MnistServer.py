import pickle
import flask
from flask_cors import CORS
import os
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))

app = flask.Flask(__name__)
CORS(app)

modelLR = pickle.load(open(os.path.join(dir_path,"MnistLR.pkl"),"rb"))
modelKNN = pickle.load(open(os.path.join(dir_path,"MnistKNN.pkl"),"rb"))
modelRF = pickle.load(open(os.path.join(dir_path,"MnistRF.pkl"),"rb"))
modelET = pickle.load(open(os.path.join(dir_path,"MnistET.pkl"),"rb"))

def getModel(key):
    switcher={
        'lr': modelLR,
        'knn': modelKNN,
        'rf': modelRF,
        'et': modelET
    }
    return switcher.get(key)

@app.route('/predict', methods=['POST'])
def predict():
    alg = flask.request.args.get('alg')
    model = getModel(alg)
    features = flask.request.get_json(force=True)
    probabilities = model.predict_proba([features])
    prediction = np.argmax(probabilities, axis=1)
    response = {'prediction': int(prediction),'probabilities':probabilities[0].tolist()}
    return flask.jsonify(response)

app.run(host='127.0.0.1', port=5500)