
#%%
import numpy as np
import scipy as sp 

from IPython.display import display
from matplotlib import pyplot as plt
import matplotlib as mpl

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D # TODO - is this import realy neccessary? it recognize 1D pooling withouy it
from keras.layers.normalization import BatchNormalization
#from keras.utils.visualize_util import plot
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras import regularizers

from DeepCNN_Functions import scheduler#LearningRate #TODO - a better way?

# def learning_rate_func(initAlpha=0.01, factor=0.5, dropEvery=10,epoch):
#     # compute the learning rate for the current epoch:
#     exp = np.floor((1 + epoch)/self.dropEvery)
# 	alpha = self.initAlpha*(self.factor**exp) 
#     return float(alpha)

''' 1) Import the data set :''' 
#from keras.datasets import cifar10
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
from scipy.io import loadmat 
DataBase = loadmat('16PQDs_4800_WithNoise.mat')
X = [None]
Y = [None]

for i in range (0,76799): #TODO - make parametric..
#for keys in DataBase
    #temp = DataBase ['SignalsDataBase'][0][i]['signals']
    X.append(DataBase ['SignalsDataBase'][0][i]['signals'][0])
    Y.append(DataBase ['SignalsDataBase'][0][i]['labels'][0])
# t = np.linspace(0,0.2,640)
# plt.plot(t,X[6])
# plt.show()

#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = np.array(X[1:70000])

X_train = np.expand_dims(X_train,axis=-1)
numOfFeatures = X_train.shape[1]

Y_train = np.reshape( np.array(Y[1:70000]) , (69999,1) )
X_validate = np.array(X[70000:len(X)])

X_validate = np.expand_dims(X_validate,axis=-1)

Y_validate =  np.reshape( np.array(Y[70000:len(Y)]) ,(6800,1) )


#%%
''' 2) Normalize the data and Onehot encoding  '''

# Make OneHot classes :
from sklearn.preprocessing import OneHotEncoder#,LabelEncoder
OHE = OneHotEncoder()
#y_test_OneHot = OHE.fit_transform(y_test).toarray()
Y_train_OneHot = OHE.fit_transform(Y_train).toarray()
Y_validate_OneHot = OHE.fit_transform(Y_validate).toarray()


#%%
#  Train a Deep convolutional neural network '''

# Build the CNN model :
model = keras.models.Sequential() #TODO : how to write model = Sequential instead

#add model layers:
'''######################################### UNIT1 #############################################'''
# Conv1:
model.add(keras.layers.Conv1D(filters = 32, kernel_size=3,strides = 1, activation='relu', input_shape=(numOfFeatures,1)))
# Conv2:
model.add(keras.layers.Conv1D(filters = 32, kernel_size=3,strides = 1, activation='relu'))
# Pool1:
model.add(keras.layers.MaxPool1D(pool_size=(3), strides = 1)) #TODO : data_format='channels_first' ??
# BN1:
model.add(keras.layers.BatchNormalization())
'''########################################## UNIT2 #####################################'''
# Conv3:
model.add(keras.layers.Conv1D(filters = 64, kernel_size=3,strides = 1, activation='relu'))
# Conv4:
model.add(keras.layers.Conv1D(filters = 64, kernel_size=3,strides = 1, activation='relu'))
# Pool2:
model.add(keras.layers.MaxPool1D(pool_size=(3), strides = 1)) #TODO : data_format='channels_first' ??
# BN2:
model.add(keras.layers.BatchNormalization())
'''########################################### UNIT3 ####################################'''
# Conv5:
model.add(keras.layers.Conv1D(filters = 128, kernel_size=3,strides = 1, activation='relu'))
# Conv6:
model.add(keras.layers.Conv1D(filters = 128, kernel_size=3,strides = 1, activation='relu'))
# Pool3:
model.add(keras.layers.GlobalMaxPooling1D()) #TODO : data_format='channels_first' ??
# BN4:
model.add(keras.layers.BatchNormalization())

'''########################################### Fuuly Connected Part ####################################'''

model.add(tf.keras.layers.Flatten()) 
# Dense1:
model.add(keras.layers.Dense(units=256, activation='relu',use_bias=True))
# Dense2:
model.add(keras.layers.Dense(units=128, activation='relu',use_bias=True))
# BN3:
model.add(keras.layers.BatchNormalization())
# Dense3:
model.add(keras.layers.Dense(units=16, activation='softmax',use_bias=True)) #TODO : kernel_regularizer=regularizers.l2(0.1) ?

# training parameters:

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
                                                   
model.compile(  loss        = 'categorical_crossentropy',
                optimizer   = 'nadam',
                metrics     = ['accuracy']   
             )

# Train the model:
model_history = model.fit(X_train, Y_train_OneHot, 
                        batch_size = 64,
                        epochs=43 , 
                        callbacks=[callback],
                        validation_data=(X_validate, Y_validate_OneHot),
                        verbose=1)

# Plot CNN's accuracy vs. epoch num:
fig = plt.figure()
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN : Accuracy vs. number of Epochs')
plt.legend(['train','validation'])
plt.show()

'''Save model and weights'''
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("_MiniBatch_lr_db4800_'withNoise.h5")
print("Saved model to disk")

####################################################################################
''' $$$$$$$$$$$$$$$$$$$$ INFERENCE: $$$$$$$$$$$$'''

# load json and create model:
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# load weights into new model:
loaded_model.load_weights("model_MiniBatch_lr_db4800_'withNoise.h5")
print("Loaded model from disk")

# evaluate loaded model on test data

loaded_model.compile(loss        = 'categorical_crossentropy',
                    optimizer   = 'nadam',
                    metrics     = ['accuracy']   
                    )

score = loaded_model.evaluate(X_validate, Y_validate_OneHot, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))



''' Prediction : '''

# Given a signal with a disturb 
Xtest = np.array(X_validate[62:63])
Ytest = Y_validate[62:63]
#Xtest = np.expand_dims(Xtest,axis=-1)

XtestPredict = np.round(loaded_model.predict(Xtest))
Xtest_label = OHE.inverse_transform(XtestPredict)[0][0]
verdict = (Xtest_label == Ytest[0])
print("Prediction:", Xtest_label , ", Label:" , Ytest[0][0],"\nDeepCNN prediction is",verdict[0])

plt.plot(Xtest[0])
plt.show()

# ###### EOF
