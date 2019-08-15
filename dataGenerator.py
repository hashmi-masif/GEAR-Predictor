from keras.utils import to_categorical
from PIL import Image
import pandas as pd
import numpy as np
from dataPreparation import *

p = np.random.permutation(len(df))
train_up_to = int(len(df) * TRAIN_TEST_SPLIT)
train_idx = p[:train_up_to]
test_idx = p[train_up_to:]

# split train_idx further into training and validation set
train_up_to = int(train_up_to * 0.7)
train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

df['gender_id'] = df['gender'].map(lambda gender: GENDER_ID_MAP[gender])
df['race_id'] = df['race'].map(lambda race: RACE_ID_MAP[race])

max_age = df['age'].max()
len(train_idx), len(valid_idx), len(test_idx), max_age


def get_data_generator(df, indices, for_training, batch_size=16):
    images, ages, races, genders = [], [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, age, race, gender = r['file'], r['age'], r['race_id'], r['gender_id']
            im = Image.open(file)
            im = im.resize((IM_WIDTH, IM_HEIGHT))
            im = np.array(im) / 255.0
            images.append(im)
            
            if(int(age)>60):
                ages.append(to_categorical(6,7))
            elif(int(age)<=60 and int(age)>50):
                ages.append(to_categorical(5,7))
            elif(int(age)<=50 and int(age)>40):
                ages.append(to_categorical(4,7))
            elif(int(age)<=40 and int(age)>30):
                ages.append(to_categorical(3,7))
            elif(int(age)<=30 and int(age)>25):
                ages.append(to_categorical(2,7))
            elif(int(age)<=25 and int(age)>20):
                ages.append(to_categorical(1,7))
            elif(int(age)<=20):
                ages.append(to_categorical(0,7))
                
            
            races.append(to_categorical(race, len(RACE_ID_MAP)))
            genders.append(to_categorical(gender, 2))
            if len(images) >= batch_size:
                yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]
                images, ages, races, genders = [], [], [], []
        if not for_training:
            break