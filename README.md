# GEAR Predictor

GEAR stand for Gender, Emotion, Age, Race predictor. This is a deep learning project that predict the GEAR of a preson using the image of their face. First it detect all faces present in the image and then using deep learning model it predict the GEAR of the respective images and makes a directory with the annotated images as output.

##  Prerequisites

Before running the project follow the following steps

1. Get all the required files and libraries from the following command `` pip install -r requirements.txt``
2. System must have a Nvidia GPU with cuda installed and tensorflow with GPU enabled

##  Getting Started

To run the software follow the steps

1. Place any number of images inside the images directory if it is of high resolution or have a dimension greater than 700 x 700 else place the image in the imagesLow directory.
2. Run the software using the following command if the image size is of low resolution or less than 700 x 700 with ``python runGEARlow.py`` else run the following command `` python runGEAR.py``
3. Output annotated images will be in the updated_images and extracted faces will also be present in faces directory

##  Training the models

To train the model on your own use the file AgeGenderRaceTraining.py file or to change the model architecture use the file AgeGenderRaceModel.py

##  Build with

* [KERAS](https://www.keras.io/) - Deep learning framework used
* [OPENCV](https://www.opencv.org/) - Opencv for face detection
* [UTKFace dataset](https://www.kaggle.com/jangedoo/utkface-new) - UTK face dataset for age, gender, race detection
* [Fer2013](https://www.kaggle.com/deadskull7/fer2013) - For emotion detection
* [FACE DATASET](https://www.kaggle.com/deadskull7/fer2013) - For testing age, ethnicity and emotion

## Input

![](/images/3men.jpg)

## Output

![](/updated_images/fgs.jpg)

## Author

* **Mohammad Asif Hashmi** - [Udolf15](https://github.com/Udolf15)

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/Udolf15/GEAR-Predictor/blob/master/LICENSE) file for details

