# -*- coding: utf-8 -*-
"""
June 2024
@author: jwhitebored
github: https://github.com/jwhitebored/polypal.git

NOTE: I'm currently looking for work in data science, python development, or
      math-related fields. If you're hiring, shoot me an email at
      james.white2@mail.mcgill.ca
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn as skl
from tensorflow import keras
from tensorflow.keras import layers

#"If you have issues running on GPU, paste these two lines:"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

######################### Data via CSV ########################################
number_of_batches = 1
batch_start = 26

for i in range (number_of_batches):
    path = 'C:/Users/'
    df = pd.read_csv(path + '50k_and_10k_signals_batch_' + str(i+batch_start) + '.csv', float_precision='round_trip')
    #pdf = pd.read_csv(path + '50k_and_10k_polys_batch_' + str(i+batch_start) + '.csv', float_precision='round_trip') #These are the polynomial graphs without noise
    coefdf = pd.read_csv(path + 'coefficients_batch_' + str(i+batch_start) + '.csv', float_precision='round_trip')
    polycoeffs = coefdf.iloc[:,2:].to_numpy()
    
    if (i==0):
        polydegs = df.iloc[:,0].to_numpy()
        signals = df.iloc[:,2:].to_numpy()
        #polys = pdf.iloc[:,2:].to_numpy()
        polycoeffs = coefdf.iloc[:,2:].to_numpy()
        
    else:
        polydegs = np.concatenate((polydegs, df.iloc[:,0].to_numpy()), axis=0)
        signals = np.concatenate((signals, df.iloc[:,2:].to_numpy()), axis=0)
        #polys = np.concatenate((polys, pdf.iloc[:,2:].to_numpy()), axis=0)
        polycoeffs = np.concatenate((polycoeffs, coefdf.iloc[:,2:].to_numpy()), axis=0)

maxdeg = 10# Quite the question of how many degrees I should train the AI on.
maxcoeff = 10 #Also quite arbitrary. Perhaps there is some scholarly source on the frequency of certain polynomial coeffs
numpolys = 60000
train_num = 50000
test_num = numpolys-train_num
signallength=1024

######################### Normalize Data ######################################
from sklearn.preprocessing import normalize

(x_train, y_train), (x_test, y_test) = ((np.array(signals[:train_num]), np.array(polydegs[:train_num])),
                                        (np.array(signals[train_num:train_num+test_num]), np.array(polydegs[train_num:train_num+test_num])))

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train = normalize(x_train, norm="l1")
x_test = normalize(x_test, norm="l1")

x_train = x_train.reshape((50000, 32, 32))
x_test = x_test.reshape((10000, 32, 32))

############################# LSTM MODEL ######################################
model = keras.Sequential()
model.add(keras.Input(shape=(None, 32)))
model.add(layers.LSTM(256, return_sequences=True, activation='tanh', kernel_initializer='he_normal'))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(256, return_sequences=True, activation='tanh', kernel_initializer='he_normal'))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(256, return_sequences=True, activation='tanh', kernel_initializer='he_normal'))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(256, return_sequences=True, activation='tanh', kernel_initializer='he_normal'))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(256, return_sequences=True, activation='tanh', kernel_initializer='he_normal'))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(256, activation='tanh', kernel_initializer='he_normal'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10))
print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
]

#model.fit(x_train, y_train, batch_size=32, epochs=1000, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)
#model = tf.keras.models.load_model(path_model + 'model.keras')

########################## Save Model in chunks ###############################
path_model = 'C:/Users/'
num_loops=10
loop_epoch_num=10   #the number of epochs run in a loop
loop_start_num =0   #where the loop was left off at the end of the last round
for i in range(num_loops):
    training = model.fit(x_train, y_train, batch_size=64, epochs=loop_epoch_num, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)
    evaluation = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
    
    model.save(path_model+str(loop_start_num+loop_epoch_num+loop_epoch_num*i)+'epochs_test'+str(int(evaluation[1]*100))+'_train'+str(int(training.history['accuracy'][-1]*100))+'_loss'+str(int(evaluation[0]*100))+'_batch64_normL1_netL6_batchNum26_noisySignal.keras')
    print("Loops completed:" + str(i))
    
####################### Calculating Confusion Matrix ##########################

from sklearn.metrics import confusion_matrix

y_pred = model.predict(x_test)
y_pred_prob = tf.nn.softmax(y_pred, axis=-1).numpy()
y_pred_labels = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred_labels)

# Print or visualize the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

############### Comparing Wrong Predictions to The Correct Values #############

error_index = []
for i in range(len(y_test)):
    if y_pred_labels[i] != y_test[i]:
        error_index = error_index + [i]
    else:
        pass

testcoeffs = polycoeffs[train_num:]
mask = y_pred_labels != y_test
error_prob = y_pred_prob[mask]
comparison_coeffs = testcoeffs[mask]
error_y_pred_labels = y_pred_labels[mask]
error_y_true_labels = y_test[mask]

plt.hist(error_y_pred_labels)
plt.title("Predicted Poly Degrees")
plt.hist(error_y_true_labels)
plt.title("True Poly Degrees")
    
i=4
print("\n")
print(f"Prediction Prob {error_prob[i][error_y_pred_labels[i]]} True Prob {error_prob[i][error_y_true_labels[i]]}")
print(f"Prediction {error_y_pred_labels[i]} True {error_y_true_labels[i]}")
print(comparison_coeffs[i])
    
    
    
    
    