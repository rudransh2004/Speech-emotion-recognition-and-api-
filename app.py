from flask import Flask, render_template, request, redirect
import tensorflow as tf
import keras 
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from keras.models import model_from_json
import librosa
import numpy as np
import os
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def mfcc(file):
  y,sr  = librosa.load(file, duration=3,offset = 0.5)
  mfcc_feature=np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
  return mfcc_feature

json_file = open('emotion.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("emotion.h5")
print("model loaded")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
  

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        file = request.files['file']
        
        if 'file' not in request.files:
            flash('No file')
        if file.filename == '':
            flash('No selected file')
            
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'],
                secure_filename(file.filename))
            file.save(filename)
        with tf.device("cpu"):
            test = mfcc(filename)
            print(test)
            test = np.array(test)
            test = np.expand_dims(test,0) 
            test = loaded_model.predict(test)
            x = np.argmax(test,axis=-1)
            if x == [0]:
                result = "angry"
            if x ==[1]:
                result = "disgust"
            if x ==[3]:
                result = "happy"
            if x ==[4]:
                result = "neutral"
            if x ==[5]:
                result = "pleasant suprised"
            if x ==[6]:
                result = "sad"
        return {"result":result}  
           
        
        


   


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=False)