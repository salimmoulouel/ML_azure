from flask import Flask, render_template, request

from scipy.misc import imread, imresize
import numpy as np
import tensorflow 
from tensorflow.keras.models import model_from_json
from tensorflow.keras.backend import  set_session
import tensorflow
import re
import base64

#creer une sesion pour permmettre les calculs du graph
sess=tensorflow.Session()
#creer le graph de calcul keras
graph = tensorflow.get_default_graph()

#toujours utiliser la session lors du chargement du model et lors de la prediction aussi        
set_session(sess)
with open("MNIST_models/Conv_model/convolutional_model.json") as json_file:
    model=model_from_json(json_file.read())

#charger les poids du model
model.load_weights("MNIST_models/Conv_model/convolutional_weights.h5")
#compiler le model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


app = Flask(__name__)



def stringToTimaget(img):
    """detecte si une image est en format base64 dans un string et la convertit en format acceptable par le model 28x28"""
    imgstr=re.search(r'base64,(.*)',str(img)).group(1)
    with open('image.png','wb') as output:
        output.write(base64.b64decode(imgstr))
@app.route("/")
def index():
    """page root"""
    return render_template("index.html")

@app.route("/predict/",methods=['POST'])
def predict():
    """prediction du chiffre à partir de l'appel à l'API
    curl -X POST -F img=@six.png http://localhost:9999/predict/
    """

    global model, graph
    #get data recupere les données jointe a la requete post
    imgData= request.get_data()
    try:
        stringToImage(imgData)
    except:
        f = request.files['img']
        f.save('image.png')
    x=imread('image.png',mode='L')
    x = imresize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)
    with graph.as_default():
        #preciser que c'est la session qui a été enregistrée precedement et qui contient le modéle instantié
        set_session(sess)
        prediction = model.predict(x)
        response = np.argmax(prediction,axis=1)
        return str(response[0])+"\n"




if __name__=="__main__":
    app.run(host='0.0.0.0', port=9200)
    #retirer le debug lors de la mise en production
    app.run(debug=True)