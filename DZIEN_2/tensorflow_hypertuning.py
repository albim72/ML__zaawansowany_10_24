import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner import RandomSearch

#definicja modelu realizaowana za pomocą funkcji
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers',min_value=1,max_value=3)):
        model.add(layers.Flatten(input_shape=(28,28)))
        model.add(layers.Dense(units=hp.Int('units'+str(i),min_value=32,max_value=512,step=32),
                               activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(
        hp.Choice('learning_rate',
                  values=[1e-2,1.5e-3,1.7e-4,2.2e-5])),
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy']
    )
    return model

    
