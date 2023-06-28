##############################################3
        #IMPORTACION DE PAQUETES
##############################################
import os
import pandas as pd
import cv2
import imutils
import numpy as np
import tensorflow as tf
import math



#DEFINICION DE LAS FUNCIONES PARA PREPROCESAMIENTO
#DEL DATASET
#--------------------------------------------
def subdirectorios(dir):
    rutas=[]
    for i in os.listdir(dir):
        ruta=os.path.join(dir,i)
        if ruta[0]=='.':
           ...
        else:
            rutas.append(ruta)
    rutas.sort()
    return rutas

def crear_dataframe(dir):
    dir_img,dir_label=subdirectorios(dir)
    rutas_imagenes=subdirectorios(dir_img)
    rutas_labels=subdirectorios(dir_label)
    dataframe=pd.DataFrame({'imagenes':rutas_imagenes,'labels':rutas_labels})
    return dataframe


def leer_bboxes(fila):
    with open(fila,'r') as f:
        texto=f.read()

        lista=texto.split('\n')
        for elem in lista:
            if elem=='':
                lista.remove(elem)
        return lista


def eliminar(fila, valor):
    for i in fila:
        if i==valor:
            fila.remove(valor)
    return fila 



def separar_boxes(dataframe):
    imagenes=[]
    labels=[]
    for i in range(dataframe.shape[0]):
        img=dataframe['imagenes'][i]
        label=dataframe['boxes'][i]
        for j in label:
            imagenes.append(img)
            labels.append(j)
    return pd.DataFrame({'imagenes':imagenes,'labels':labels})


def obtener_elem(fila,indice):
    if fila[indice]==' ':
        print('fnd')
    return float(fila[indice])

def append_index_using_best_iou(x, anchors):
    for bbox in x:
        x1_true, y1_true, x2_true, y2_true, _ = bbox
        w_true = x2_true - x1_true
        h_true = y2_true - y1_true
        index = -1
        max_iou = -1
        for ind, anchor in enumerate(anchors.tolist()):
            h_anch, w_anch = anchor
            intersection = min(w_true, w_anch) * min(w_true, w_anch)
            union = (w_true * h_true) + (w_anch * h_anch) - intersection
            iou = intersection / union
            if iou > max_iou:
                max_iou = iou
                index = ind
        bbox.append(index)
    return x
def comprobar_tam(dir,ALTO):
    i=0
    while i<1:
        for elem in os.listdir(f'{dir}/images'):
            im_ej=cv2.imread(f'{dir}/images/{elem}')
            i+=1
    if ALTO!=im_ej.shape[0]:
        for elem in os.listdir(f'{dir}/images'):
            im_ej=cv2.imread(f'{dir}/images/{elem}')
            im_ej=imutils.resize(im_ej,width=ALTO,height=ALTO)
            os.remove(f'{dir}/images/{elem}')
            cv2.imwrite(f'{dir}/images/{elem}' , im_ej)
    else:
        print('el tamaño es correcto')


def preparacion_dataframe(dir_train,dir_val,ANCHORS,ALTO):
    #Comprobacion de los datos
    comprobar_tam(dir_train,ALTO)
    comprobar_tam(dir_val,ALTO)

    #Creacion del dataframe
    datos_train=crear_dataframe(dir_train)
    datos_val=crear_dataframe(dir_val)

    #Extraer las cajas de la ruta
    datos_train['boxes']= datos_train['labels'].apply(lambda x: leer_bboxes(x) )
    datos_train=datos_train.drop(columns='labels')
    datos_val['boxes']=datos_val['labels'].apply(lambda x: leer_bboxes(x) )
    datos_val=datos_val.drop(columns='labels')

    #Eliminar valores no deseados
    datos_train['boxes']=datos_train['boxes'].apply(lambda x: eliminar(x,''))
    datos_val['boxes']=datos_val['boxes'].apply(lambda x: eliminar(x,''))

    #Separar los diferents box de la misma imagen
    datos_train=separar_boxes(datos_train)
    datos_val=separar_boxes(datos_val)

    #Comprobamos que no hay valores no deseados
    datos_train['labels']= datos_train['labels'].apply(lambda x: x.split(' '))
    datos_train['labels']= datos_train['labels'].apply(lambda x: eliminar(x,''))
    datos_val['labels']= datos_val['labels'].apply(lambda x: x.split(' '))
    datos_val['labels']= datos_val['labels'].apply(lambda x: eliminar(x,''))

    #Separa los bbox en  clase, x, y, w, h
    datos_train['clase']=datos_train['labels'].apply(lambda x:obtener_elem(x,0))
    datos_train['x']=datos_train['labels'].apply(lambda x:obtener_elem(x,1))
    datos_train['y']=datos_train['labels'].apply(lambda x:obtener_elem(x,2))
    datos_train['w']=datos_train['labels'].apply(lambda x:obtener_elem(x,3))
    datos_train['h']=datos_train['labels'].apply(lambda x:obtener_elem(x,4))
    datos_val['clase']=datos_val['labels'].apply(lambda x:obtener_elem(x,0))
    datos_val['x']=datos_val['labels'].apply(lambda x:obtener_elem(x,1))
    datos_val['y']=datos_val['labels'].apply(lambda x:obtener_elem(x,2))
    datos_val['w']=datos_val['labels'].apply(lambda x:obtener_elem(x,3))
    datos_val['h']=datos_val['labels'].apply(lambda x:obtener_elem(x,4))
    
    #Cambiamos los elem a clase, xmin, ymin, xmax, ymax
    datos_train['xmin']=datos_train.apply(lambda x: x['x']-(x['w']/2.0),axis=1)
    datos_train['ymin']=datos_train.apply(lambda x: x['y']-(x['h']/2.0),axis=1)
    datos_train['xmax']=datos_train.apply(lambda x: x['x']+(x['w']/2.0),axis=1)
    datos_train['ymax']=datos_train.apply(lambda x: x['y']+(x['w']/2.0),axis=1)
    datos_val['xmin']=datos_val.apply(lambda x: x['x']-(x['w']/2.0),axis=1)
    datos_val['ymin']=datos_val.apply(lambda x: x['y']-(x['h']/2.0),axis=1)
    datos_val['xmax']=datos_val.apply(lambda x: x['x']+(x['w']/2.0),axis=1)
    datos_val['ymax']=datos_val.apply(lambda x: x['y']+(x['w']/2.0),axis=1)

    #Eliminamos las columnas que ya no se necesitan
    datos_train=datos_train.drop(columns=['x','y','w','h'])
    datos_val=datos_val.drop(columns=['x','y','w','h'])

    #Unimos los valores de los puntos en un columnas bbox
    datos_train['bboxes']=datos_train.apply(lambda x: [x[3],x[4],x[5],x[6],x[2]],axis=1)
    datos_train= datos_train.drop(columns=['xmin', 'ymin', 'xmax', 'ymax', 'clase'])
    datos_train=datos_train.groupby('imagenes').agg({'bboxes':list})
    datos_train['imagenes']=list(datos_train.index)
    datos_train.reset_index(drop=True,inplace=True)

    datos_val['bboxes']=datos_val.apply(lambda x: [x['xmin'], x['ymin'], x['xmax'], x['ymax'], x['clase']],axis=1)
    datos_val= datos_val.drop(columns=['xmin', 'ymin', 'xmax', 'ymax', 'clase'])
    datos_val=datos_val.groupby('imagenes').agg({'bboxes':list})
    datos_val['imagenes']=list(datos_val.index)
    datos_val.reset_index(drop=True,inplace=True)

    #Append index
    datos_train['bboxes']=datos_train['bboxes'].apply(lambda x: append_index_using_best_iou(x, ANCHORS))
    datos_val['bboxes']=datos_val['bboxes'].apply(lambda x: append_index_using_best_iou(x, ANCHORS))

    return datos_train,datos_val

#### DEFINICION DEL PIPELINE PARA EL DATABASE ###
#-------------------------------------------
class Pipeline(tf.keras.utils.Sequence):
    def __init__(self,dataframe,xcol,ycol,batch_size,img_h,img_w,img_c,num_clases,grid_sizes,anchors,anchor_masks):
        #Preparacion de las variables
        self.dataframe=dataframe
        self.xcol=xcol
        self.ycol=ycol
        self.batch_size=batch_size
        self.img_h=img_h
        self.img_w=img_w
        self.img_c = img_c
        self.num_clases=num_clases,
        self.grid_sizes=grid_sizes
        self.anchors=anchors
        self.anchor_masks=anchor_masks

    def __len__(self):
        return int(self.dataframe.shape[0]/self.batch_size)
    
    def on_epoch_end(self):
        self.dataframe=self.dataframe.sample(frac=1)
        self.dataframe.reset_index(drop=True,inplace=True)
    
    def __getitem__(self,item):
        X=np.zeros(shape=(self.batch_size,self.img_h,self.img_w,self.img_c),dtype='float32')
        Y20=np.zeros(shape=(self.batch_size,self.grid_sizes[0],self.grid_sizes[0],len(self.anchor_masks[0]),4+1+1+4),dtype='float32')
        Y40=np.zeros(shape=(self.batch_size,self.grid_sizes[1],self.grid_sizes[1],len(self.anchor_masks[1]),4+1+1+4),dtype='float32')
        Y80=np.zeros(shape=(self.batch_size,self.grid_sizes[2],self.grid_sizes[2],len(self.anchor_masks[2]),4+1+1+4),dtype='float32')

        for i in range(self.batch_size):
            ruta_im=self.dataframe[self.xcol][i+(self.batch_size*item)]
            im=cv2.imread(ruta_im)
            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            im =np.array(im,dtype='float32')
            im=im/255.0
            X[i,:,:,:]=im
            bboxes=self.dataframe[self.ycol][i+(self.batch_size*item)]
            for box in bboxes:
                x1,y1,x2,y2,c,indice=box
                x=(x1+x2)/2.0
                y=(y1+y2)/2.0
                w=x2-x1
                h=y2-y1
                if indice in self.anchor_masks[0]:
                    grid_x=math.floor(self.grid_sizes[0]*x)
                    grid_y=math.floor(self.grid_sizes[0]*y)
                    Y20[i,grid_y,grid_x,indice %len(self.anchor_masks[0]),:]=[x1,y1,x2,y2,1.0,c,x,y,w,h]
                
                elif indice in self.anchor_masks[1]:
                    grid_x=math.floor(self.grid_sizes[1]*x)
                    grid_y=math.floor(self.grid_sizes[1]*y)
                    Y40[i,grid_y,grid_x,indice %len(self.anchor_masks[1]),:]=[x1,y1,x2,y2,1.0,c,x,y,w,h]
                
                elif indice in self.anchor_masks[2]:
                    grid_x=math.floor(self.grid_sizes[2]*x)
                    grid_y=math.floor(self.grid_sizes[2]*y)
                    Y80[i,grid_y,grid_x,indice %len(self.anchor_masks[2]),:]=[x1,y1,x2,y2,1.0,c,x,y,w,h]
                





        return tf.convert_to_tensor(X,dtype='float32'),[tf.convert_to_tensor(Y20,dtype='float32'),tf.convert_to_tensor(Y40,dtype='float32'),tf.convert_to_tensor(Y80,dtype='float32')]
    
    

    
