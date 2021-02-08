from flask import Flask
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import os 
import base64
import re
import cv2

# variables Flask
app = Flask(__name__)
api = Api(app)


# se carga el modelo de Logistic Regression del Notebook #3
dir_path = os.path.dirname(os.path.realpath(__file__))
pkl_filename = "ModeloLR.pkl"
with open(os.path.join(dir_path,pkl_filename), 'rb') as file:
    model = pickle.load(file)

Mnist_filename = "MnistLR.pkl"
with open(os.path.join(dir_path,Mnist_filename), 'rb') as file:
    MnistModel = pickle.load(file)

def convertImage(imgData1):
        imgData1 = imgData1.decode("utf-8")
        imgstr = re.search(r'base64,(.*)',imgData1).group(1)
        #print(imgstr)
        imgstr_64 = base64.b64decode(imgstr)
        with open('output/output.png','wb') as output:
                output.write(imgstr_64)

class Predict(Resource):

    @staticmethod
    def post():
        # parametros
        parser = reqparse.RequestParser()
        parser.add_argument('petal_length')
        parser.add_argument('petal_width')
        parser.add_argument('sepal_length')
        parser.add_argument('sepal_width')

        # request para el modelo
        args = parser.parse_args() 
        datos = np.fromiter(args.values(), dtype=float) 

        # prediccion
        out = {'Prediccion': int(model.predict([datos])[0])}

        return out, 200

    @staticmethod
    def get():
        parser = reqparse.RequestParser()
        parser.add_argument('petal_length')
        parser.add_argument('petal_width')
        parser.add_argument('sepal_length')
        parser.add_argument('sepal_width')

        # request para el modelo
        args = parser.parse_args() 
        datos = np.fromiter(args.values(), dtype=float) 

        # prediccion
        out = {'Prediccion': int(model.predict([datos])[0])}

        return out, 200
api.add_resource(Predict, '/predict')

class Predict_Number(Resource):
    @staticmethod
    def get():
        parser = reqparse.RequestParser()
        parser.add_argument('ImageBase64')
        args = parser.parse_args() 
        datosDecoded = base64.b64decode(args.ImageBase64)
        datosDecoded = np.frombuffer(datosDecoded, dtype=np.uint8)
        if not cv2.imwrite('InvertedOutput.png', datosDecoded.reshape((28, 28))):
            raise Exception("Could not write file")
        out = {'Prediccion': int(MnistModel.predict([datosDecoded])[0])}
        return out, 200
api.add_resource(Predict_Number, '/predict-number')


if __name__ == '__main__':
    app.run(debug=True, port='1080')