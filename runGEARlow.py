# Importing required modules and library
import os
import cv2
import numpy as np
from dataPreparation import * 
from PIL import Image
from keras.models import load_model


# Loading required trained models
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
modelFinal =load_model('trainedModels/modelAgeRaceGenderV4.h5')
modelFinal2 =load_model('trainedModels/eModel2.h5')
modelFinal.summary()

# Define paths
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'trainedModels/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + 'trainedModels/weights.caffemodel')
EMOTIONS_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

import numpy as np


# Create directory 'updated_images' if it does not exist
if not os.path.exists('updated_images'):
	print("New directory created")
	os.makedirs('updated_images')

# Loop through all images and save images with marked faces
for file in os.listdir(base_dir + 'imagesLow'):
	file_name, file_extension = os.path.splitext(file)
	if (file_extension in ['.png','.jpg','.jpg.jpeg']):
		print("Image path: {}".format(base_dir + 'images/' + file))

		image = cv2.imread(base_dir + 'imagesLow/' + file)

		(h, w) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

		model.setInput(blob)
		detections = model.forward()
		font                   = cv2.FONT_HERSHEY_SIMPLEX
		fontScale              = .5
		fontColor              = (255,0,0)
		lineType               = 1

		# Create frame around face
		for i in range(0, detections.shape[2]):
			# Extracting the frames for each face saving them and predicting the age, emotion, race and gender
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			bottomLeftCornerOfText = (startX-30,endY+ 1)
			confidence = detections[0, 0, i, 2]

			# If confidence > 0.5, show box around face
			if (confidence > 0.5):
				#extract
				frame = image[startY-10:endY, startX-10:endX+10]
				frame = cv2.resize(frame,(64,64))
				cv2.imwrite(base_dir + 'faces/' + 'test' + '_' + file, frame)

				# Saving the images to faces directory
				im1 = Image.open(base_dir + 'faces/' + 'test' + '_' + file)
				# Reading the images
				im2 = cv2.imread(base_dir + 'faces/' + 'test' + '_' + file)
				im1 = im1.resize((198, 198))
				im1 = np.array(im1) / 255.0
				images = []

				# Converting the 3 channel rgb image to 1 channel grayscale image
				im2 = cv2.cvtColor(im2,cv2.COLOR_RGB2GRAY)
				roi = cv2.resize(im2, (64, 64))
				roi = np.array(roi) / 255.0
				images.append(im1)
				imO = np.array(images)

				preds = modelFinal.predict(imO)
				preds2 = modelFinal2.predict(roi[np.newaxis, :, :, np.newaxis])
				age = ID_AGE_MAP[preds[0].argmax(axis = -1)[0]]
				race = ID_RACE_MAP[preds[1].argmax(axis = -1)[0]]
				gender = ID_GENDER_MAP[preds[2].argmax(axis = -1)[0]]
				emotion = EMOTIONS_LIST[preds2[0].argmax(axis = -1)]

				# Making the box around image and writing the predictions
				cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)
				print(str(age)+', '+str(race)+', '+str(gender)+', '+str(emotion))
				cv2.putText(image,str(age)+', '+str(race)+', '+str(gender)+', '+str(emotion), 
							bottomLeftCornerOfText, 
							font, 
							fontScale,
							fontColor,
							lineType)

		cv2.imwrite(base_dir + 'updated_images/' + file, image)
		print("Image " + file + " converted successfully")