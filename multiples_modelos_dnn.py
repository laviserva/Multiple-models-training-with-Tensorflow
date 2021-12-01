# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:17:08 2021

@author: Lavi
"""
#librerias para deep learning
import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, InputLayer, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

import sklearn
from sklearn.metrics import confusion_matrix

#Librerias para graficar
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import csv #crear csv
import os #manejar paths por el sistema operativo
import datetime #fechas y horas
import numpy as np
import math
import time #manejar tiempos de reloj

from create_data import create_images #librería para importar archivo pickle

lr = 5.5e-4
EPOCHS = 150
#data = create_images(pickle_file = "esc_01.pickle")
data = create_images(pickle_file = r"pick")
BATCH_SIZE = 256
technique = "weight" #Oversampling, undersampling o pesos -> weight
filtros_conv = [128,64,32,16,8,4]
capas_conv = np.arange(len(filtros_conv)) + 1
dense_neuronas = [64,32,10,5]
dense_layers = np.arange(len(dense_neuronas)) +1
Regularizacion = [False, True ]

weight_decay = 1e-4


######################################### Funciones de apoyo #########################################
def step_decay(epoch):
    """
    Cambia el learning rate por cada iteracion. Por lo cual, se pueden iniciar con learnings rates "altos"
    """
	initial_lr = 5.5e-4
	drop = 0.99
	epochs_drop = 1
	lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lr

def plot_cm(labels1, predictions, labels2, predictions_train, name, p=0.5):    
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(name)
    
    cm1 = confusion_matrix(labels2, predictions_train > p)
    
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm1.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm1.flatten()/np.sum(cm1)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    ax1 = plt.subplot(221)
    sns.heatmap(cm1, annot=labels, fmt='')
    ax1.set_title('Training Set')
    ax1.set_ylabel('Actual label')
    ax1.set_xlabel('Predicted label')
    
    cm2 = confusion_matrix(labels1, predictions > p)
    
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    cm2.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm2.flatten()/np.sum(cm2)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    ax2 = plt.subplot(222)
    sns.heatmap(cm2, annot=labels, fmt='')
    ax2.set_title('Validation Set')
    ax2.set_ylabel('Actual label')
    ax2.set_xlabel('Predicted label')
    
    fig_name = os.path.join("cm_graphs",name_dnn)
    plt.savefig(fig_name)
    print(fig_name)    

######################################### Cargar datos para colores #########################################
mpl.rcParams['figure.figsize'] = (15, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
######################################### Cargar datos de entrenamiento #########################################
[train_images, train_labels] = data

split = 17
div = int(len(train_labels) * split / 100)

test_images = train_images[:div]
test_labels = train_labels[:div]

train_images = train_images[div:]
train_labels = train_labels[div:]

bool_train_labels = train_labels != 0

del(split)
del(div)

######################################### pesos de clase #########################################
neg = len(train_labels)
pos = 0
for i in train_labels:
    if i == 1:
        pos += 1
neg -= pos
del(i)

initial_bias = np.log([pos/neg])

pesos_para_0 = (1 / neg) * ((pos + neg) / 2.0)
pesos_para_1 = (1 / pos) * ((pos + neg) / 2.0)

class_weight = {0: pesos_para_0, 1: pesos_para_1}

print('Weight for class 0: {:.2f}'.format(pesos_para_0))
print('Weight for class 1: {:.2f}'.format(pesos_para_1))

######################################### Establecer paths y crear direcotrios #########################################
h5_dirs = "h5_logs"
tensorboard_dir = "Tensorboard_logs"
csv_dir = "csv_logs"
images_path = "cm_graphs"
weights_path = "weights"

try:
    os.mkdir(os.path.join(os.getcwd(), h5_dirs))
except:
    print("Ya existe el path ", os.path.join(os.getcwd(), h5_dirs))
try:
    os.mkdir(os.path.join(os.getcwd(), tensorboard_dir))
except:
    print("Ya existe el path ", os.path.join(os.getcwd(), tensorboard_dir))
try:
    os.mkdir(os.path.join(os.getcwd(), csv_dir))
except:
    print("Ya existe el path ", os.path.join(os.getcwd(), csv_dir))
try:
    os.mkdir(os.path.join(os.getcwd(), images_path))
except:
    print("Ya existe el path ", os.path.join(os.getcwd(), images_path))
try:
    os.mkdir(os.path.join(os.getcwd(), weights_path))
except:
    print("Ya existe el path ", os.path.join(os.getcwd(), weights_path))
    
    
######################################### Establecer Callbacks #########################################
class myCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.7):
        super(myCallback, self).__init__()
        self.threshold = threshold
        self.best_macrof1 = 0.994
        
    def on_epoch_end(self, epoch, logs={}):
        
        global acc_log
        acc_log = logs.get('accuracy')
        
        val_fp = logs.get("val_fp")
        val_fn = logs.get("val_fn")
        
        val_presicion = logs.get("val_precision")
        val_recall = logs.get("val_recall")
        
        presicion = (logs.get("precision") + val_presicion)/2
        recall = (logs.get("recall") + val_recall)/2
        
        macr_f1 = (2*presicion*recall) / (presicion+recall)
        
        if macr_f1 > self.best_macrof1:
            self.best_macrof1 = macr_f1

            nombre = os.path.join(h5_dirs,name_dnn) + "_val_fp" + str(val_fp) + "_val_fn" + str(val_fn) + ".h5" 
            #self.model.stop_training = True
            full_path = os.path.join(os.getcwd(),nombre)
            #print(full_path)
            try:
                modelo.save(full_path)
            except:
                pass

callbacks = myCallback() #Cargando callback, se puede agregar threshold = x | 0<x<1

h5_path = os.path.join(os.getcwd(),h5_dirs)
tensorboard_path =  os.path.join(os.getcwd(),tensorboard_dir)
csv_path =  os.path.join(os.getcwd(),csv_dir)

######################################### Establecer modelos para la IA #########################################
opt = tf.keras.optimizers.Adam(lr=lr)

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]
######################################### Sobremuestreo #########################################
if technique.lower() == "oversampling":
    bool_train_labels = train_labels != 0
    
    pos_features = train_images[bool_train_labels]
    neg_features = train_images[~bool_train_labels]
    
    pos_labels = train_labels[bool_train_labels]
    neg_labels = train_labels[~bool_train_labels]
    
    ids = np.arange(len(pos_features))
    choices = np.random.choice(ids, len(neg_features))
    
    res_pos_features = pos_features[choices]
    res_pos_labels = pos_labels[choices]
    
    resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
    resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)
    
    order = np.arange(len(resampled_labels))
    np.random.shuffle(order)
    resampled_features = resampled_features[order]
    resampled_labels = resampled_labels[order]
    
    split = 17
    div = int(len(resampled_labels) * split / 100)
    
    test_images = resampled_features[:div]
    test_labels = resampled_labels[:div]
    
    train_images = resampled_features[div:]
    train_labels = resampled_labels[div:]
    
    del(split)
    del(div)

######################################### Undersampling #########################################
if technique.lower == "undersampling":

    bool_train_labels = train_labels != 0
    
    pos_features = train_images[bool_train_labels]
    neg_features = train_images[~bool_train_labels]
    
    pos_labels = train_labels[bool_train_labels]
    neg_labels = train_labels[~bool_train_labels]
    
    neg_features = neg_features[:len(pos_labels)]
    neg_labels = neg_labels[:len(pos_labels)]
    
    resampled_features = np.concatenate([pos_features, neg_features], axis=0)
    resampled_labels = np.concatenate([pos_labels, neg_labels], axis=0)
    
    order = np.arange(len(resampled_labels))
    np.random.shuffle(order)
    
    resampled_features = resampled_features[order]
    resampled_labels = resampled_labels[order]
    
    split = 17
    div = int(len(resampled_labels) * split / 100)
    
    test_images = resampled_features[:div]
    test_labels = resampled_labels[:div]
    
    train_images = resampled_features[div:]
    train_labels = resampled_labels[div:]
    
    del(split)
    del(div)

######################################### CSV preparation #########################################
csv_rows = []
csv_dnn_name = []
f1_score = []
val_f1_score = []
macro_f1_score = []
macro_precision = []
macro_recall = []
time_fit = []
time_predict = []
    

######################################### Entrenamiento #########################################
for regularizacion in Regularizacion:
    for capa_conv in capas_conv:
        if regularizacion == True:
            name_dnn = "SP_R_B"
        if regularizacion == False:
            name_dnn = "SP_B"
            
        modelo = Sequential()
        modelo.add(InputLayer(input_shape=(1,280,1)))
            
        for conv_index in range(len(filtros_conv)):
            if conv_index >= capa_conv:
                continue
            if conv_index < capa_conv:
                if regularizacion == False:
                    name_dnn += "_C" + str(filtros_conv[conv_index])
                    modelo.add(Conv2D(filtros_conv[conv_index],(1,3)))
                    modelo.add(BatchNormalization())
                    
                if regularizacion == True:
                    name_dnn += "_C" + str(filtros_conv[conv_index])
                    modelo.add(Conv2D(filtros_conv[conv_index],(1,3), kernel_regularizer=tf.keras.regularizers.l2((weight_decay))))
                    modelo.add(BatchNormalization())
                    modelo.add(Dropout(0.3))
                
        modelo.add(Flatten())
        
        for dense_index, capa_dense in enumerate(dense_neuronas):
            
            if dense_index >= len(dense_layers):continue
            
            if dense_index == 0:
                
                if regularizacion == False:
                    
                    name_dnn += "_D" + str(capa_dense) + "_D1"
                    modelo.add(Dense(capa_dense, activation='relu'))
                    modelo.add(Dense(1, activation='sigmoid'))
                    
                if regularizacion == True:
                    
                    name_dnn += "_D" + str(capa_dense) + "_D1"
                    modelo.add(Dense(capa_dense, activation='relu', kernel_regularizer=regularizers.l2((weight_decay))))
                    modelo.add(Dense(1, activation='sigmoid'))
            
            
            if dense_index < len(dense_layers) and dense_index != 0:
                
                modelo.pop()
                name_dnn = name_dnn[:-3]
                
                if regularizacion == False:
                    
                    name_dnn += "_D" + str(capa_dense) + "_D1"
                    modelo.add(Dense(capa_dense, activation='relu'))
                    modelo.add(Dense(1, activation='sigmoid'))
                    
                if regularizacion == True:
                    
                    name_dnn += "_D" + str(capa_dense) + "_D1"
                    modelo.add(Dense(capa_dense, activation='relu' ,kernel_regularizer=regularizers.l2((weight_decay))))
                    modelo.add(Dense(1, activation='sigmoid'))
            modelo.summary()
            #print(" ###################################################################################################### ")
            
            try:
                os.mkdir(os.path.join(h5_path, name_dnn))
                print("Se ha creado el path: ", os.path.join(h5_path, name_dnn))
            except:
                print("Ya existe el path ", os.path.join(h5_path, name_dnn))
            try:
                os.mkdir(os.path.join(tensorboard_path, name_dnn))
                print("Se ha creado el path: ", os.path.join(tensorboard_path, name_dnn))
            except:
                print("Ya existe el path ", os.path.join(tensorboard_path, name_dnn ))
            try:
                os.mkdir(os.path.join(csv_path, name_dnn))
                print("Se ha creado el path: ", os.path.join(csv_path, name_dnn))
            except:
                print("Ya existe el path ", os.path.join(csv_path, name_dnn))
            
            
            tensorboard_log = os.path.join(tensorboard_path, name_dnn)
            log_dir = os.getcwd() + "\Tensorboard_logs\\" + "acc_" + str(acc_log) + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= tensorboard_log, histogram_freq=1)

            h5_dir = os.getcwd() + r"\h5_models" + "\Save_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
            h5_log = os.path.join(h5_path, name_dnn)
            checkpoint_path = os.path.join(h5_log, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath= checkpoint_path,
                save_weights_only=True,
                save_freq=len(train_labels),
                verbose = 1)
            
            csv_file= os.path.join(csv_path, name_dnn + ".csv" )
            
            #Creando weights iniciales para repetir resultados
            initial_weights = os.path.join(os.getcwd(), weights_path )
            
            weights_update = os.path.join(initial_weights, name_dnn + ".h5")
            
            if os.path.isfile(weights_update) == True: modelo.load_weights(weights_update)
            elif os.path.isfile(weights_update) == False: modelo.save_weights(weights_update)
                
            modelo.layers[-1].bias.assign(initial_bias)
            modelo.compile(loss = "binary_crossentropy", optimizer = opt, metrics = METRICS )
            time_init = time.time()            

            ######################################### Oversampling #########################################

            if technique.lower() == "oversampling":
                modelo_con_pesos = modelo.fit(
                    resampled_ds,
                    validation_data=(val_ds),
                    # These are not real epochs
                    steps_per_epoch=20,
                    epochs=EPOCHS,
                    callbacks=[callbacks, tensorboard_callback,
                                               model_checkpoint_callback,
                                               CSVLogger(csv_file),
                                               LearningRateScheduler(step_decay)])
            ######################################### Submuestreo #########################################          
            elif technique.lower() == "undersampling":
                modelo_con_pesos = modelo.fit(train_images, train_labels, epochs= EPOCHS,
                                    validation_data = (test_images , test_labels),
                                    batch_size = BATCH_SIZE,
                                    callbacks=[callbacks, tensorboard_callback,
                                               model_checkpoint_callback,
                                               CSVLogger(csv_file), LearningRateScheduler(step_decay, verbose=1)])
                
            ######################################### Pesos #########################################
            else:
                modelo_con_pesos = modelo.fit(train_images, train_labels, epochs= EPOCHS,
                                    validation_data = (test_images , test_labels),
                                    batch_size = BATCH_SIZE,
                                    class_weight=class_weight,
                                    callbacks=[callbacks, tensorboard_callback,
                                               model_checkpoint_callback,
                                               CSVLogger(csv_file), LearningRateScheduler(step_decay, verbose=1)])
           
            modelo.save_weights(h5_log + ".h5")
            time_init = time.time() - time_init
            apoyo = []
            
            for i in range(len(modelo_con_pesos.history.keys())): apoyo.append(list(modelo_con_pesos.history.values())[i][-1])
            
            csv_rows.append(apoyo)
            csv_dnn_name.append(name_dnn)
            numerador = 2 * modelo_con_pesos.history["precision"][-1] * modelo_con_pesos.history['recall'][-1]
            denominador = modelo_con_pesos.history["precision"][-1] + modelo_con_pesos.history['recall'][-1]
            val_numerador = 2 * modelo_con_pesos.history["val_precision"][-1] * modelo_con_pesos.history['val_recall'][-1]
            val_denominador = modelo_con_pesos.history["val_precision"][-1] + modelo_con_pesos.history['val_recall'][-1]
            f1_score.append(numerador / denominador)
            val_f1_score.append(val_numerador / val_denominador)
            
            macro_precision.append((modelo_con_pesos.history["precision"][-1] + modelo_con_pesos.history["val_precision"][-1]) / 2)
            macro_recall.append((modelo_con_pesos.history['recall'][-1] + modelo_con_pesos.history['val_recall'][-1]) / 2)
            
            time_fit.append(time_init)
            
            time_init = time.time()
            train_predictions_weighted = modelo.predict(train_images, batch_size=BATCH_SIZE)
            time_init = time.time() - time_init
            test_predictions_weighted = modelo.predict(test_images, batch_size=BATCH_SIZE)
            
            time_predict.append(time_init)
            
            
            
            baseline_results = modelo.evaluate(test_images, test_labels,
                                              batch_size=BATCH_SIZE, verbose=0)
            
            plot_cm(test_labels, test_predictions_weighted, train_labels, train_predictions_weighted ,name_dnn)

######################################### Creacion csv para metricas #########################################
headers = ["nombre de la red"] + [key for key in list(modelo_con_pesos.history.keys())] + ["f1-score"] 
headers = headers + ["val_f1-score"] + ["macro_f1-score"]+ ["macro-precision"] + ["macro-recall"] + ["time-fit(s)"]
headers = headers + ["time-predict(s)"]

#convertir a numpy
f1_score = np.array(f1_score)
val_f1_score = np.array(val_f1_score)
csv_rows = np.array(csv_rows)
time_fit = np.array(time_fit)
time_predict = np.array(time_predict)
macro_precision = np.array(macro_precision)
macro_recall = np.array(macro_recall)

f1_score = np.expand_dims(f1_score, axis=0)
csv_dnn_name = np.expand_dims(np.array(csv_dnn_name), axis=0)
val_f1_score = np.expand_dims(val_f1_score, axis=0)
macro_precision = np.expand_dims(macro_precision, axis=0)
macro_recall = np.expand_dims(macro_recall, axis=0)
time_fit = np.expand_dims(time_fit, axis=0)
time_predict = np.expand_dims(time_predict, axis=0)

macro_f1_score = (f1_score + val_f1_score) / 2
#macro_f1_score = np.expand_dims(np.array(macro_f1_score), axis=0)

#Expandir matriz
csv_rows = np.concatenate((csv_dnn_name, csv_rows.T), axis=0).T
csv_rows = np.concatenate((csv_rows.T, f1_score), axis=0).T
csv_rows = np.concatenate((csv_rows.T, val_f1_score), axis=0).T
csv_rows = np.concatenate((csv_rows.T, macro_f1_score), axis=0).T
csv_rows = np.concatenate((csv_rows.T, macro_precision), axis=0).T
csv_rows = np.concatenate((csv_rows.T, macro_recall), axis=0).T
csv_rows = np.concatenate((csv_rows.T, time_fit), axis=0).T
csv_rows = np.concatenate((csv_rows.T, time_predict), axis=0).T


with open('Train_history.csv', 'w', newline='') as csvfile:
    file = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    file.writerow(headers)
    for row in csv_rows:
        file.writerow(row)
        
        
        