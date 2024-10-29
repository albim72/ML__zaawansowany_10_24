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

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='mys_dir',
    project_name='myparams'
)

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train,x_test = x_train/255.0,x_test/255.0

tuner.search(x_train,y_train,
             epochs=5,
             validation_data = (x_test,y_test))

best_model = tuner.get_best_models(num_models=1)[0]
print(tuner.results_summary())

print("OCENA MODELU:\n")
eval_loss,eval_accuracy = best_model.evaluate(x_test,y_test)
print(f'najlepszy model: Loss -> {eval_loss}, Accuracy -> {eval_accuracy}')
