from keras.callbacks import ModelCheckpoint
from dataGenerator import *
from AgeRaceGenderModel import *

#Creating the model

modelTrain = createModel(IM_WIDTH ,IM_HEIGHT)

batch_size = 64
valid_batch_size = 64
train_gen = get_data_generator(df, train_idx, for_training=True, batch_size=batch_size)
valid_gen = get_data_generator(df, valid_idx, for_training=True, batch_size=valid_batch_size)

callbacks = [
    ModelCheckpoint("./model_checkpoint", monitor='val_loss')
]

history = modelTrain.fit_generator(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=10,
                    callbacks=callbacks,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)
                    

modelTrain.save("modelAgeRaceGender.h5")