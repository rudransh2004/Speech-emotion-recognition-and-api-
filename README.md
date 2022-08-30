
# Speech Emotion Recognition using Deep learning and API 

This repository lets you develop your own speech emotion recogntion Neural network using Tensorflow and Keras
and deploy it as an API on heroku using Flask. 

# Libraries
    1) Tensorflow
    2) Keras
    3) Librosa
    4) Flask 

# Dataset
Toronto emotional speech set (TESS) dataset was used which is easily available on Kaggle. 

# Inference 

1) To Train the model and do some sort of inference on it refer to Speech_emotion_recognition_Rudransh_Agnihotri_.ipynb

2) To use this API please run app.py and start localhost:5000 on your browser 




## API Reference

#### POST

```http
 POST localhost:5000
```
Use  form-data multipart to POST audio files 
| Key | Type     
| :-------- | :------- | 
| `file` | `audio file ` 

To test API you can use Postman 

### Deployed on Heroku

link to the API https://speechapi.herokuapp.com/



## 🚀 About Me
I'm a Deep learning and ML Developer 

