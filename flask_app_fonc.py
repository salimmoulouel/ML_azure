from flask import Flask, render_template, request

from scipy.misc import imread, imresize
import numpy as np
from tensorflow.keras.models import model_from_json
import tensorflow 

json_file = open('MNIST_models/Conv_model/convolutional_model.json','r')
model_json = json_file.read()
json_file.close()

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

#creer une session et un graph de calcul!!! important 

sess = tensorflow.Session()
graph = tensorflow.get_default_graph()

#toujours utiliser la session lors du chargement du model et lors de la prediction aussi
set_session(sess)
model = model_from_json(model_json)
model.load_weights("MNIST_models/Conv_model/convolutional_weights.h5")
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,optimizer=tensorflow.keras.optimizers.Adam(),metrics=['accuracy'])

app = Flask(__name__)

@app.route('/')
def index():
    return "izan"#render_template("index.html")

import re
import base64

def stringToImage(img):
    imgstr = re.search(r'base64,(.*)', str(img)).group(1)
    with open('image.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    global model, graph
    imgData = request.get_data()
    try:
        stringToImage(imgData)
    except:
        f = request.files['img']
        f.save('image.png')
        
    x = imread('image.png', mode='L')
    x = imresize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)
    
   
    
    
    with graph.as_default():
        
        #toujours utiliser la session lors du chargement du model et lors de la prediction aussi
        set_session(sess)
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        # convert the response to a string
        response = np.argmax(out, axis=1)
        return str(response[0])+"\n"


if __name__ == "__main__":

    # run the app locally on the given port
    app.run(host='0.0.0.0', port=9000)
# optional if we want to run in debugging mode
    app.run(debug=True)