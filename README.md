
# GEAR Predictor

  

GEAR stand for Gender, Emotion, Age, Race predictor. This is a deep learning project that predict the GEAR of a preson using the image of their face. First it detect all faces present in the image and then using deep learning model it predict the GEAR of the respective images and makes a directory with the annotated images as output.

  This software can provide 2 functionalities 
	
1. Batch processing of images present in a folder with extraction of faces as well as marking the faces and predicting the GEAR properties from images.
2. This software can be used to predict the GEAR properties from webcam using videos in real time. 

## Prerequisites

  

Before running the project follow the following steps

  
1. Get all the required files and libraries from the following command `` pip install -r requirements.txt``

2. System must have a Nvidia GPU with cuda installed and tensorflow with GPU enabled

  

## Getting Started

  

To run the software follow the steps

1. Enter the following command ```  python run.py ```
2. From the 3 buttons present named as :
	1. REALTIME PROCESSING
	2. BATCH PROCESSING OF LOW QUALITY IMAGES
	3. BATCH PROCESSING OF HIGH QUALITY IMAGES
3. Select the type of processing you want from the software
4. Place Low quality images in imagesLow directory.
5. Place High quality images in images directory.
6. After batch processing the images will be placed in updated images as well as faces extracted can be found in faces directory.
7. To stop the realtime webcam please press escape key on keyboard.

![](/images/gp.PNG) 

## Training the models

  

To train the model on your own use the file AgeGenderRaceTraining.py file or to change the model architecture use the file AgeGenderRaceModel.py

  

## Build with

  

*  [KERAS](https://www.keras.io/) - Deep learning framework used

*  [OPENCV](https://www.opencv.org/) - Opencv for face detection

*  [UTKFace dataset](https://www.kaggle.com/jangedoo/utkface-new) - UTK face dataset for age, gender, race detection

*  [Fer2013](https://www.kaggle.com/deadskull7/fer2013) - For emotion detection

*  [FACE DATASET](https://www.kaggle.com/deadskull7/fer2013) - For testing age, ethnicity and emotion

  

## Input

  

![](/images/3men.jpg)

  

## Output

  

![](/update_images/fgs.jpg)

  

## Author

  

*  **Mohammad Asif Hashmi** - [Udolf15](https://github.com/Udolf15)

  

## License

  

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/Udolf15/recommendMeMovies/blob/master/LICENSE) file for details