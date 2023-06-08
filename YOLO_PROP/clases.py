import os
import cv2
import imutils
import pandas as pd
import warnings
import tensorflow as tf
import numpy as np
import math
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def comprobar_tam(dir,tam):
    for elem in os.listdir(f'{dir}/images'):
        img=cv2.imread(f'{dir}/images/{elem}')
        if tam!=img.shape[1]:
            img=imutils.resize(img,width=tam,height=tam)
            os.remove(f'{dir}/images/{elem}')
            cv2.imwrite(f'{dir}/images/{elem}',img)
    print('\n--------------\nTamaños revisados\n--------------\n')

def subdirs(dir,n):
    rutas=[]
    for elem in os.listdir(dir):
        ruta=os.path.join(dir,elem)
        #Comprobamos que no sea un directorio oculto
        if n==True:
            if ruta.count('.')>=1:
                ...
            else:
                rutas.append(ruta)
        else:
            rutas.append(ruta)
    rutas.sort()
    return rutas

def crear_dataframe(dir):
    #obtenemos los dir de imagenes y labels
    dir_img,dir_label=subdirs(dir,True)
    rutas_img=subdirs(dir_img,False)
    rutas_labels=subdirs(dir_label,False)
    dataframe=pd.DataFrame({'imagenes':rutas_img,'labels':rutas_labels})
    return dataframe

def leer_bboxes(x):
    with open(x,'r') as f:
        texto=f.read()
        lista=texto.split('\n')
        for elem in lista:
            if elem=='':
                lista.remove(elem)
        return lista

def eliminar(x,valor):
    for i in x:
        if i==valor:
            x.remove(valor)
    return x



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


def preparacion_frame(dir_entre,dir_val,anchors,tam):

    #Comprobamos que las imagenes de ambos directorios tienen el tamaño adecuado
    comprobar_tam(dir_entre,tam)
    comprobar_tam(dir_val,tam)

    #Creamos el dataframe
    datos_entre=crear_dataframe(dir_entre)
    datos_val=crear_dataframe(dir_val)
    #------Extraccion de la informacion del archivo de labels-----------
    #Extraemos las cajas a partir de la ruta de labels
    datos_entre['boxes']=datos_entre['labels'].apply(lambda x: leer_bboxes(x))
    datos_val['boxes']=datos_val['labels'].apply(lambda x: leer_bboxes(x))

    #Eliminamos la columna labels
    datos_entre=datos_entre.drop(columns='labels')
    datos_val=datos_val.drop(columns='labels')

    #Eliminamos los valores no deseados
    datos_entre['boxes']=datos_entre['boxes'].apply(lambda x: eliminar(x,''))
    datos_val['boxes']=datos_val['boxes'].apply(lambda x: eliminar(x,''))


    #Separar los diferents box de la misma imagen
    datos_train=separar_boxes(datos_entre)
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
    datos_train['bboxes']=datos_train['bboxes'].apply(lambda x: append_index_using_best_iou(x, anchors))
    datos_val['bboxes']=datos_val['bboxes'].apply(lambda x: append_index_using_best_iou(x, anchors))
    
    return datos_val,datos_train

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
        Y20=np.zeros(shape=(self.batch_size,self.grid_sizes[0],self.grid_sizes[0],len(self.anchor_masks[0]),4+1+4),dtype='float32')
        Y40=np.zeros(shape=(self.batch_size,self.grid_sizes[1],self.grid_sizes[1],len(self.anchor_masks[1]),4+1+4),dtype='float32')
        Y80=np.zeros(shape=(self.batch_size,self.grid_sizes[2],self.grid_sizes[2],len(self.anchor_masks[2]),4+1+4),dtype='float32')



        for i in range(self.batch_size):
            ruta_im=self.dataframe[self.xcol][i+(self.batch_size*item)]
            im=cv2.imread(ruta_im)
            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            im =np.array(im,dtype='float32')
            im=im/255
            X[i,:,:,:]=im
            bboxes=self.dataframe[self.ycol][i+(self.batch_size*item)]
            for box in bboxes:
                x1,y1,x2,y2,_,indice=box
                x=(x1+x2)/2.0
                y=(y1+y2)/2.0
                w=x2-x1
                h=y2-y1
                if indice in self.anchor_masks[0]:
                    grid_x=math.floor(self.grid_sizes[0]*x)
                    grid_y=math.floor(self.grid_sizes[0]*y)
                    Y20[i,grid_y,grid_x,indice %len(self.anchor_masks[0]),:]=[x1,y1,x2,y2,1.0,x,y,w,h]
                
                elif indice in self.anchor_masks[1]:
                    grid_x=math.floor(self.grid_sizes[1]*x)
                    grid_y=math.floor(self.grid_sizes[1]*y)
                    Y40[i,grid_y,grid_x,indice %len(self.anchor_masks[1]),:]=[x1,y1,x2,y2,1.0,x,y,w,h]
                
                elif indice in self.anchor_masks[2]:
                    grid_x=math.floor(self.grid_sizes[2]*x)
                    grid_y=math.floor(self.grid_sizes[2]*y)
                    Y80[i,grid_y,grid_x,indice %len(self.anchor_masks[2]),:]=[x1,y1,x2,y2,1.0,x,y,w,h]
                
        return tf.convert_to_tensor(X,dtype='float32'),[tf.convert_to_tensor(Y20,dtype='float32'),tf.convert_to_tensor(Y40,dtype='float32'),tf.convert_to_tensor(Y80,dtype='float32')]


###PREPARACION DEL MODELO
##CLASES DE POSTPROCESAMIENTO
class PostProcesor(tf.keras.layers.Layer):
  def __init__(self, num_classes, grid_size, anchors, anchor_mask):
        super(PostProcesor, self).__init__(trainable=False, dynamic=False, dtype=tf.float32)
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.anchors = anchors
        self.anchor_mask = anchor_mask
  
  def call(self, inputs,training): # (B, G, G, A, 5 + N)
        print(inputs.shape)
        batch_size = tf.shape(inputs)[0] # B
        xy, wh, p, c = tf.split(inputs, [2, 2, 1, self.num_classes], axis=-1) # (B, G, G, A, 2), (B, G, G, A, 2), (B, G, G, A, 1), (B, G, G, A, N)
        print(xy.shape)
        xy = tf.math.sigmoid(xy) # (B, G, G, A, 2)
        p = tf.math.sigmoid(p) # (B, G, G, A, 1)
        #c = tf.math.softmax(c) # (B, G, G, A, N)
        c = tf.math.sigmoid(c) # (B, G, G, A, N) # USE THIS ONLY WHEN C_LOSS IS BINARY CROSSENTROPY
        xywh = tf.concat([xy, wh], axis=-1) # (B, G, G, A, 4)
        print(xywh)
        grid = tf.meshgrid(tf.range(self.grid_size), tf.range(self.grid_size)) # [(G, G), (G, G)]
        grid = tf.stack(grid, axis=-1) # (G, G, 2)
        grid = tf.expand_dims(grid, axis=-2) # (G, G, 1, 2)
        grid = tf.expand_dims(grid, axis=0) # (1, G, G, 1, 2)
        grid = tf.tile(grid, [batch_size, 1, 1, len(self.anchor_mask), 1]) # (B, G, G, A, 2)
        print(grid.shape,'\n',xy.shape)
        grid = tf.cast(grid, dtype=tf.float32) # (B, G, G, A, 2)
        xy = (xy + grid) / tf.cast(self.grid_size, dtype=tf.float32) # (B, G, G, A, 2)
        anch = tf.constant(self.anchors[self.anchor_mask], dtype=tf.float32) # (A, 2)
        anch = tf.expand_dims(anch, axis=0) # (1, A, 2)
        anch = tf.expand_dims(anch, axis=0) # (1, 1, A, 2)
        anch = tf.expand_dims(anch, axis=0) # (1, 1, 1, A, 2)
        anch = tf.tile(anch, [batch_size, self.grid_size, self.grid_size, 1, 1]) # (B, G, G, A, 2)
        wh = tf.math.exp(wh) * anch # (B, G, G, A, 2)
        x1y1 = xy - (wh / 2.0) # (B, G, G, A, 2)
        x2y2 = xy + (wh / 2.0) # (B, G, G, A, 2)
        bbox = tf.concat([x1y1, x2y2], axis=-1) # (B, G, G, A, 4)
        return bbox, p, c, xywh

class NMS(tf.keras.layers.Layer):
    def __init__(self, num_classes, iou_threshold, score_threshold, max_bboxes):
        super(NMS, self).__init__(trainable=False, dynamic=False, dtype=tf.float32)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_bboxes = max_bboxes
        self.reshape_b20 = tf.keras.layers.Reshape(target_shape=(-1, 4))
        self.reshape_p20 = tf.keras.layers.Reshape(target_shape=(-1, 1))
        self.reshape_c20 = tf.keras.layers.Reshape(target_shape=(-1, self.num_classes))
        self.reshape_b40 = tf.keras.layers.Reshape(target_shape=(-1, 4))
        self.reshape_p40 = tf.keras.layers.Reshape(target_shape=(-1, 1))
        self.reshape_c40 = tf.keras.layers.Reshape(target_shape=(-1, self.num_classes))
        self.reshape_b80 = tf.keras.layers.Reshape(target_shape=(-1, 4))
        self.reshape_p80 = tf.keras.layers.Reshape(target_shape=(-1, 1))
        self.reshape_c80 = tf.keras.layers.Reshape(target_shape=(-1, self.num_classes))
        self.bbox_concat = tf.keras.layers.Concatenate(axis=1)
        self.p_concat = tf.keras.layers.Concatenate(axis=1)
        self.c_concat = tf.keras.layers.Concatenate(axis=1)
        self.reshape = tf.keras.layers.Reshape(target_shape=(-1, 1, 4))

    def call(self, inputs, training): # [[(B, 13, 13, 3, 4), (B, 13, 13, 3, 1), (B, 13, 13, 3, N)], [(B, 26, 26, 3, 4), (B, 26, 26, 3, 1), (B, 26, 26, 3, N)], [(B, 52, 52, 3, 4), (B, 52, 52, 3, 1), (B, 52, 52, 3, N)]]
        [bbox20, p20, c20], [bbox40, p40, c40], [bbox80, p80, c80] = inputs # [(B, 13, 13, 3, 4), (B, 13, 13, 3, 1), (B, 13, 13, 3, N)], [(B, 26, 26, 3, 4), (B, 26, 26, 3, 1), (B, 26, 26, 3, N)], [(B, 52, 52, 3, 4), (B, 52, 52, 3, 1), (B, 52, 52, 3, N)]
        bbox20 = self.reshape_b20(bbox20) # (B, 13 * 13 * 3, 4)
        p20 = self.reshape_p20(p20) # (B, 13 * 13 * 3, 1)
        c20 = self.reshape_c20(c20) # (B, 13 * 13 * 3, N)
        bbox40 = self.reshape_b40(bbox40) # (B, 26 * 26 * 3, 4)
        p40 = self.reshape_p40(p40) # (B, 26 * 26 * 3, 1)
        c40 = self.reshape_c40(c40) # (B, 26 * 26 * 3, N)
        bbox80 = self.reshape_b80(bbox80) # (B, 52 * 52 * 3, 4)
        p80 = self.reshape_p80(p80) # (B, 52 * 52 * 3, 1)
        c80 = self.reshape_c80(c80) # (B, 52 * 52 * 3, N)
        bbox = self.bbox_concat([bbox20, bbox40, bbox80]) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3), 4)
        p = self.p_concat([p20, p40, p80]) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3), 1)
        c = self.c_concat([c20, c40, c80]) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3), N)
        c = tf.argmax(c, axis=-1) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3))
        c = tf.cast(c, dtype=tf.float32) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3))
        c = tf.expand_dims(c, axis=-1) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3), 1)
        score = p * c # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3), 1)
        bbox = self.reshape(bbox) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3), 1, 4)
        bbox, score, c, valid = tf.image.combined_non_max_suppression(boxes=bbox, scores=score, max_output_size_per_class=self.max_bboxes, max_total_size=self.max_bboxes, iou_threshold=self.iou_threshold, score_threshold=self.score_threshold) # (B, M, 4), (B, M), (B, M), (B,)
        score = tf.expand_dims(score, axis=-1) # (B, M, 1)
        c = tf.expand_dims(c, axis=-1) # (B, M, 1)
        pred = tf.concat([bbox, score, c], axis=-1) # (B, M, 4 + 1 + 1)
        return pred, valid

class nms1(tf.keras.layers.Layer):
    def __init__(self, num_classes, iou_threshold, score_threshold, max_bboxes):
        super(nms1, self).__init__(trainable=False, dynamic=False, dtype=tf.float32)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_bboxes = max_bboxes
        self.reshape_b20 = tf.keras.layers.Reshape(target_shape=(-1, 4))
        self.reshape_p20 = tf.keras.layers.Reshape(target_shape=(-1, 1))
        self.reshape_c20 = tf.keras.layers.Reshape(target_shape=(-1, self.num_classes))
        self.reshape = tf.keras.layers.Reshape(target_shape=(-1, 1, 4))
    
    def call(self, inputs, training):
        [bbox20, p20, c20]=inputs
        bbox20 = self.reshape_b20(bbox20) # (B, 13 * 13 * 3, 4)
        p20 = self.reshape_p20(p20) # (B, 13 * 13 * 3, 1)
        c20 = self.reshape_c20(c20) # (B, 13 * 13 * 3, N)
        c = tf.argmax(c20, axis=-1) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3))
        c = tf.cast(c, dtype=tf.float32) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3))
        c = tf.expand_dims(c, axis=-1)
        score = p20 * c # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3), 1)
        bbox = self.reshape(bbox20) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3), 1, 4)
        bbox, score, c, valid = tf.image.combined_non_max_suppression(boxes=bbox, scores=score, max_output_size_per_class=self.max_bboxes, max_total_size=self.max_bboxes, iou_threshold=self.iou_threshold, score_threshold=self.score_threshold) # (B, M, 4), (B, M), (B, M), (B,)
        score = tf.expand_dims(score, axis=-1) # (B, M, 1)
        c = tf.expand_dims(c, axis=-1) # (B, M, 1)
        pred = tf.concat([bbox, score, c], axis=-1) 
        return pred, valid
##CLASES PARA LAS DISTINTAS CAPAS
##DEFINICION DEL BLOQUE DE CONVOLUCION
class CONV(tf.keras.layers.Layer):
  def __init__(self,kernel_size,strides,filters):
    super(CONV,self).__init__()
    self.kernel_size=kernel_size
    self.strides=strides
    self.filters=filters
    self.conv=tf.keras.layers.Conv2D(self.filters,self.kernel_size,self.strides,padding='same')
    self.norm=tf.keras.layers.BatchNormalization()
    self.leaky=tf.keras.layers.LeakyReLU()
  
  def call(self,inputs):
    
    x=self.conv(inputs)
    x=self.norm(x)
    x=self.leaky(x)
    return x

##DEFINICION DEL BLOQUE SPPF
class SPPF(tf.keras.layers.Layer):
  def __init__(self,filters):
    super(SPPF,self).__init__()
    self.filters=filters
    self.conv=tf.keras.layers.Conv2D(self.filters,(1,1),(1,1),padding='same')
    self.max_pool=tf.keras.layers.MaxPooling2D((2,2),(1,1),padding='same')
    self.concat=tf.keras.layers.Concatenate(axis=-1)
  
  def call(self, inputs):
    x=self.conv(inputs)
    x,x1=tf.split(x,2,axis=-1)
    x=self.max_pool(x)
    x,x2=tf.split(x,2,axis=-1)
    x=self.max_pool(x)
    x,x3=tf.split(x,2,axis=-1)
    x=self.max_pool(x)
    x=self.concat([x,x1,x2,x3])
    x=self.conv(x)
    return x

##DEFINICION DEL BLOQUE BOTTLENECK
class BOTTLENECK(tf.keras.layers.Layer):
  def __init__(self,filters,shortcut):
    super(BOTTLENECK, self).__init__()
    self.filters=filters
    self.shortcut=shortcut
    self.conv1=CONV((3,3),(1,1),0.5*self.filters)
    self.conv2=CONV((3,3),(1,1),self.filters)
  
  def call(self,inputs):
    x=self.conv1(inputs)
    x=self.conv2(x)
    #if self.shortcut==True:
      #x=inputs+x
    return x

#DEFINICION DEL BLOQUE C2F
class C2F(tf.keras.layers.Layer):
  def __init__(self,filters,num_bottles,shortcut):
    super(C2F,self).__init__()
    self.filters = filters
    self.shortcut=shortcut
    self.num_bottles=num_bottles
    self.conv=CONV((1,1),(1,1),self.filters)
    self.bottle1=BOTTLENECK(self.filters/2.0,self.shortcut)
    self.bottle2=BOTTLENECK(self.filters/2.0,self.shortcut)
    self.concat=tf.keras.layers.Concatenate(axis=-1)
  
  def call(self, inputs):
    x=self.conv(inputs)
    x,x1=tf.split(x,2,axis=-1)
    x=self.bottle1(x)
    x=self.concat([x1,x])
    x=self.conv(x)
    return x

#DEFINICION DEL BLOQUE UPSAMPPLE
class upsamp(tf.keras.layers.Layer):
  def __init__(self,filters):
    super(upsamp,self).__init__()
    self.filters=filters
    self.up=tf.keras.layers.UpSampling2D(size=(2,2))
    self.conv=tf.keras.layers.Conv2D(self.filters/2.0,(3,3),(1,1),padding='same')
  
  def call(self,inputs,conv):
    x=self.up(inputs)
    if conv==True:
      x=self.conv(x)
    return x


class OUTPUT(tf.keras.layers.Layer):
   def __init__(self,filters,num_clases,grid_size,anchor):
      super(OUTPUT,self).__init__(trainable=True, dynamic=False,dtype=tf.float32)
      self.filters=filters
      self.num_clases=num_clases
      self.grid_size=grid_size
      self.anchor=anchor

      self.conv=CONV((3,3),(1,1),self.filters)
      self.out=tf.keras.layers.Conv2D(len(self.anchor)*(5+self.num_clases),kernel_size=1)
      self.reshape=tf.keras.layers.Reshape(target_shape=(self.grid_size,self.grid_size,len(self.anchor),5+self.num_clases))
    
   def call(self,inputs,training):
      x=self.conv(inputs)
      x=self.out(x)
      x=self.reshape(x)

      return x

#DEFINICION DEL MODELO
class YOLO(tf.keras.Model):
  def __init__(self,num_clases,grid_sizes,anchors,anchor_mask,iou_threshold,score_threshold,max_bboxes):
    super(YOLO,self).__init__()
    self.num_clases=num_clases
    self.grid_sizes=grid_sizes
    self.anchors=anchors
    self.anchor_mask=anchor_mask
    self.iou=iou_threshold
    self.score_threshold=score_threshold
    self.max_bboxes=max_bboxes
    self.conv1=CONV((3,3),(2,2),64) #sale (1,320,320,64)
    self.conv2=CONV((3,3),(2,2),128) # (1,160,160,128)
    self.c2f_1=C2F(128,3,True) #No cambia el tamaño
    self.conv3=CONV((3,3),(2,2),256) #(1,80,80,256)
    self.c2f_2=C2F(256,3,True)
    self.conv4=CONV((3,3),(2,2),512) #(1,40,40,512)
    self.c2f_3=C2F(512,3,True)
    self.conv5=CONV((3,3),(2,2),512) #(1,20,20,512)
    self.c2f_4=C2F(512,3,True)
    self.sppf=SPPF(512) #No varian las dimensiones
    self.c2f_5=C2F(512,3,False)
    self.concat=tf.keras.layers.Concatenate(axis=-1)
    self.up1=upsamp(512) #(1,80,80,256)
    self.c2f_6=C2F(512,3,False)
    self.up2=upsamp(512) #(1,40,40,512)
    self.conv6=CONV((3,3),(2,2),256) #(1,40,40,256)
    self.c2f_7=C2F(512,3,False)
    self.conv7=CONV((3,3),(2,2),512)
    self.c2f_8=C2F(512,3,False)
    #SEGUIR
    self.post20=PostProcesor(2,self.grid_sizes[0],self.anchors,self.anchor_mask[0])
    #self.post40=PostProcesor(2,self.grid_sizes[1],self.anchors,self.anchor_mask[1])
    #self.post80=PostProcesor(2,self.grid_sizes[2],self.anchors,self.anchor_mask[2])
    #nms
    self.nms=nms1(self.num_clases,self.iou,self.score_threshold,self.max_bboxes)
    self.out20=OUTPUT(512,2,20,self.anchors[0:3])
    #self.out40=OUTPUT(512,2,40,self.anchors(self.anchor_mask[1]))
    #self.out80=OUTPUT(256,2,80,self.anchors(self.anchor_mask[2]))

  def call(self,inputs,training=False):
    #Rama 20x20
    x=self.conv1(inputs) #(1,320,320,64)
    x=self.conv2(x) #(1,160,160,128)
    x=self.c2f_1(x) #(1,160,160,128)
    x=self.conv3(x)
    x1=self.c2f_2(x)
    x=self.conv4(x1)
    x2=self.c2f_3(x)
    x=self.conv5(x2)
    x=self.c2f_4(x)
    x3=self.sppf(x)
    x=self.c2f_8(x3)
    x20=self.out20(x)
    if training==True:
        return x20
    bbox20, p20, c20, _ = self.post20(x20,False)
    pred, valid = self.nms([bbox20, p20, c20],False)

    return pred, valid
  
  def summary(self, img_h, img_w, img_c, batch_size):
        X_inp = tf.keras.Input(shape=(img_h, img_w, img_c), batch_size=batch_size)
        X = self.call(X_inp)
        m = tf.keras.Model(inputs=X_inp, outputs=X)
        return m.summary()


  

