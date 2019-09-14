from keras.models import load_model
import cv2
import os

EMOTIONS_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def modelLoader():
    # Loading required trained models
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    modelFinal =load_model('trainedModels/modelAgeRaceGenderV4.h5')
    emotionModel =load_model('trainedModels/eModel2.h5')
    modelFinal.summary()

    # Define paths
    prototxt_path = 'trainedModels/deploy.prototxt'
    caffemodel_path = 'trainedModels/weights.caffemodel'


    # Read the model for opencv
    model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    return (faceCascade, emotionModel, model, modelFinal)