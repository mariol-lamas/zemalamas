
from models.yolo import YOLO
from models.utils.preprocesing import Pipeline, preparacion_dataframe
import numpy as np
import cv2
import tensorflow as tf
import os
import pandas as pd
import warnings
from models.utils.layer_types import CONV, C2f, Output, PostProcessor, NMS,CustomCallback, CustomLoss
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#DATOS INCIALES PARA EL MODELO
NUM_CLASES=2
GRID_SIZES=[20,40,80]
ANCHORS=[[116.0, 90.0], [156.0, 198.0], [373.0, 326.0], [30.0, 61.0], [62.0, 45.0], [59.0, 119.0], [10.0, 13.0], [16.0, 30.0], [33.0, 23.0]]
ANCHORS = np.array(ANCHORS, dtype='float32') / np.array([640, 640], dtype='float32')
ANCHOR_MASKS=[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
IOU,THRESHOLD=0.5,0.5
MAX_BBOXES=10
BATCH_SIZE=1
TAM_MOD=640
CANALES=3
EPOCHS=20

#DATASET PARA EL ENTRENAMIENTO

dir_entre='/home/zelenza/Desktop/yl_drn/dataset/train'
dir_val='/home/zelenza/Desktop/yl_drn/dataset/valid'

print('\nCargando ...\n')
datos_val,datos_train=preparacion_dataframe(dir_entre,dir_val,ANCHORS,TAM_MOD)
print('dataframe ready')
#datos_train=datos_train.iloc[:10,:] #bloquea el entreno a 10 filas de datos
datos_tr=Pipeline(datos_train,'imagenes','bboxes', BATCH_SIZE,TAM_MOD,TAM_MOD,CANALES,NUM_CLASES,GRID_SIZES,ANCHORS,ANCHOR_MASKS)
print('Pipeline train ready')
datos_vl=Pipeline(datos_val,'imagenes','bboxes', BATCH_SIZE,TAM_MOD,TAM_MOD,CANALES,NUM_CLASES,GRID_SIZES,ANCHORS,ANCHOR_MASKS)
print('Pipeline val ready')


#ENTRENAMIENTO DEL MODELO
model=YOLO(num_classes=NUM_CLASES,
           grid_sizes=GRID_SIZES,
           anchors=ANCHORS,
           anchor_masks=ANCHOR_MASKS,
           iou_threshold=IOU,
           score_threshold=THRESHOLD,
           max_bboxes=MAX_BBOXES)

model.summary(img_h=TAM_MOD,
              img_w=TAM_MOD,
              img_c=CANALES,
              batch_size=BATCH_SIZE)

loss1=CustomLoss(NUM_CLASES,GRID_SIZES[0],ANCHORS,ANCHOR_MASKS[0],0.5)
loss2=CustomLoss(NUM_CLASES,GRID_SIZES[1],ANCHORS,ANCHOR_MASKS[1],0.5)
loss3=CustomLoss(NUM_CLASES,GRID_SIZES[2],ANCHORS,ANCHOR_MASKS[2],0.5)
loss=[loss1,loss2,loss3]


#Compilacion del modelo

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
              loss=loss,
              metrics=['accuracy'])

print('Compilacion del modelo correcta')
callbacks = [CustomCallback(datos_val, 'imagenes', 'bboxes', TAM_MOD, TAM_MOD, CANALES, NUM_CLASES, GRID_SIZES, ANCHORS, ANCHOR_MASKS, MAX_BBOXES)]
callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=3, factor=0.3))
#Entrenamiento del modelo

model.fit(datos_tr,epochs=EPOCHS, callbacks=callbacks)

model.evaluate(datos_val, batch_size=2,verbose=2)
model.save('/home/runs',save_format='h5')



#Para abrir un modelo en h5

#model=keras.models.load_model('archivo.h5')


