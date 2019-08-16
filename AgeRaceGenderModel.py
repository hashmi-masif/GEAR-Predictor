from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from keras.optimizers import SGD
from keras.models import Model

def conv_block(inp, filters=32, bn=True, pool=True):
    _ = Conv2D(filters=filters, kernel_size=3, activation='relu')(inp)
    if bn:
        _ = BatchNormalization()(_)
    if pool:
        _ = MaxPool2D()(_)
    return _

def createModel(IM_HEIGHT, IM_WIDTH):
    
    input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))

    out = conv_block(input_layer, filters=32, bn=False, pool=False)
    out = conv_block(out , filters=32*2)
    out = conv_block(out , filters=32*3)
    out = conv_block(out , filters=32*4)
    out = conv_block(out , filters=32*5)
    out = conv_block(out , filters=32*6)


    bottleneck = GlobalMaxPool2D()(out)

    # for age calculation
    out = Dense(units=256, activation='relu')(bottleneck)
    age_output = Dense(units=7, activation='softmax', name='age_output')(out)

    # for race prediction
    out = Dense(units=128, activation='relu')(bottleneck)
    race_output = Dense(units=5, activation='softmax', name='race_output')(out)

    # for gender prediction
    out = Dense(units=128, activation='relu')(bottleneck)
    gender_output = Dense(units=2, activation='softmax', name='gender_output')(out)

    model = Model(inputs=input_layer, outputs=[age_output, race_output, gender_output])
    model.compile(optimizer='rmsprop', 
                loss={'age_output': 'categorical_crossentropy', 'race_output': 'categorical_crossentropy', 'gender_output': 'categorical_crossentropy'},
                loss_weights={'age_output': 1.2, 'race_output': 1.5, 'gender_output': 1.},
                metrics={'age_output': 'accuracy', 'race_output': 'accuracy', 'gender_output': 'accuracy'})


    # Summary of the model           
    model.summary()

    return model