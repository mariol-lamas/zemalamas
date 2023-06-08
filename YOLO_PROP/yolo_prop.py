
from clases_dron import  preparacion_frame, Pipeline
from clases_prop import YOLO
import numpy as np
import tensorflow as tf
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#DATOS INCIALES PARA EL MODELO
NUM_CLASES=2
GRID_SIZES=[20,40,80]
ANCHORS=[[116.0, 90.0], [156.0, 198.0], [373.0, 326.0], [30.0, 61.0], [62.0, 45.0], [59.0, 119.0], [10.0, 13.0], [16.0, 30.0], [33.0, 23.0]]
ANCHORS = np.array(ANCHORS, dtype='float32') / np.array([640, 640], dtype='float32')
ANCHOR_MASKS=[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
IOU,THRESHOLD=0.5,0.5
MAX_BBOXES=10
BATCH_SIZE=8
TAM_MOD=640
CANALES=3
EPOCHS=10

#DATASET PARA EL ENTRENAMIENTO

dir_entre='/home/zelenza/Desktop/pruebasia/yl_drn/train'
dir_val='/home/zelenza/Desktop/pruebasia/yl_drn/valid'
print('\nCargando ...\n')
datos_val,datos_train=preparacion_frame(dir_entre,dir_val,ANCHORS,TAM_MOD)
datos_train=Pipeline(datos_train,'imagenes','bboxes', BATCH_SIZE,TAM_MOD,TAM_MOD,CANALES,NUM_CLASES,GRID_SIZES,ANCHORS,ANCHOR_MASKS)
datos_val=Pipeline(datos_val,'imagenes','bboxes', BATCH_SIZE,TAM_MOD,TAM_MOD,CANALES,NUM_CLASES,GRID_SIZES,ANCHORS,ANCHOR_MASKS)

print('\nDATOS LISTOS\n')
#Prueba del modelo
modelo=YOLO(NUM_CLASES,GRID_SIZES,ANCHORS,ANCHOR_MASKS,IOU,THRESHOLD,MAX_BBOXES)
modelo.summary(TAM_MOD,TAM_MOD,CANALES,BATCH_SIZE)
#Compilamos el modelo
modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
modelo.fit(datos_train,epochs=1)
print('\nModelo compilado correctamente\n')
