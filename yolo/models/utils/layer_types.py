##############################################3
        #IMPORTACION DE PAQUETES
##############################################
import tensorflow as tf
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import random as rn





#### DEFINICION DE LOS TIPOS DE CAPAS ###
#--------------------------------------------

  

#CAPA DE CONVOLUCION
class CONV(tf.keras.layers.Layer):
  def __init__(self,kernel_size,strides,filters):
    super(CONV,self).__init__()
    self.kernel_size=kernel_size
    self.strides=strides
    self.filters=filters
    self.conv=tf.keras.layers.Conv2D(filters=self.filters,kernel_size=self.kernel_size,strides=self.strides,padding='same',use_bias=False,kernel_regularizer=tf.keras.regularizers.L2(l2=53-4))
    self.norm=tf.keras.layers.BatchNormalization()
    self.leaky=tf.keras.layers.LeakyReLU()
  
  def call(self,inputs,training):
    
    x=self.conv(inputs)
    x=self.norm(x)
    x=self.leaky(x)
    return x
  
  def get_config(self):
        config = super(CONV, self).get_config()
        config.update({'filters': self.filters, 'kernel_size': self.kernel_size})
        return config

#CAPA C2F
class C2f(tf.keras.layers.Layer):
    def __init__(self,filtros,kernel_size,strides,shorcut,number_bottles):
        super(C2f,self).__init__(trainable=True,dynamic=False,dtype='float32')
        self.filtros=filtros
        self.kernel_size=kernel_size
        self.strides=strides
        self.n=number_bottles
        self.shortcut=shorcut
        self.conv=CONV(self.kernel_size,self.strides,self.filtros)
        self.btllnck=Bottleneck(self.filtros,self.kernel_size,self.strides,self.shortcut)
        self.conc=tf.keras.layers.Concatenate(axis=-1)
    
    def call(self,inputs):
        x=self.conv(inputs)
        x1,x=tf.split(x,2,axis=-1)
        x=self.conc([x,x1])
        x=self.conv(x)
        return x



class SPPF(tf.keras.layers.Layer):
    def __init__(self,filtros,kernel_size,strides,pool_size,strides_pooling):
        super(SPPF,self).__init__(trainable=True,dynamic=False,dtype='float32')
        self.filtros=filtros
        self.pool_size=pool_size
        self.strides_pooling=strides_pooling
        self.strides=strides
        self.kernel_size=kernel_size
        #self.conv=Conv(self.filtros,self.kernel_size,self.strides)
        self.max_pool=tf.keras.layers.MaxPooling2D(self.pool_size,self.strides_pooling,padding='same')
        self.concat=tf.keras.layers.Concatenate(axis=1)
    
    def call(self,inputs):
        x=self.conv(inputs)
        x1=x
        x=self.max_pool(x)
        x2=x
        x=self.max_pool(x)
        x3=x
        x=self.max_pool(x)
        x=self.concat([x,x3,x2,x1])
        x=self.conv(x)
        return x



#CAPA BOTTLENECK
class Bottleneck(tf.keras.layers.Layer):
    def __init__(self,filtros,kernel_size,strides,shorcut):
        super(Bottleneck,self).__init__(trainable=True,dynamic=False,dtype='float32')
        self.filtros=filtros
        self.kernel_size=kernel_size
        self.strides=strides
        self.shortcut=shorcut
        self.conv=CONV(self.kernel_size,self.strides,self.filtros)
    
    def call(self,inputs):
        if self.shortcut==True:
            x1=inputs*0.5
            x=inputs-x1
            x=self.conv(x)
            x=self.conv(x)
            return x+x1
        else:
            x=self.conv(inputs)
            x=self.conv(x)
            return x



#Capa de pooling
class Pool(tf.keras.layers.Layer):
    def __init__(self, filters, strides):
        super(Pool, self).__init__(trainable=True, dynamic=False, dtype='float32')
        self.filters = filters
        self.strides = strides
        self.pool = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=self.strides, padding='same', activation='relu')

    def call(self, inputs, training):
        return self.pool(inputs)

    def get_config(self):
        config = super(Pool, self).get_config()
        config.update({'filters': self.filters, 'strides': self.strides})
        return config


#Capa de salida
class Output(tf.keras.layers.Layer):
    def __init__(self,filtros,num_clases,grid_size,anchor):
        super(Output,self).__init__(trainable=True,dynamic=False,dtype='float32')
        self.filtros=filtros
        self.num_clases=num_clases
        self.grid_size=grid_size
        self.anchor=anchor
        self.conv=CONV((3,3),(1,1),filters=self.filtros)
        self.out=tf.keras.layers.Conv2D(len(self.anchor)*(5+self.num_clases),kernel_size=1)
        self.reshape=tf.keras.layers.Reshape(target_shape=(self.grid_size,self.grid_size,len(self.anchor),5+self.num_clases))

    def call(self,inputs,training):
        x=self.conv(inputs)
        x=self.out(x)
        x=self.reshape(x)
        return x

    def get_config(self):
        config=super(Output,self).get_config()
        config.update({'filters':self.filtros,'num_clases':self.num_clases,'grid_size':self.grid_size,'anchors':self.anchor})
        return config

#Capa de postprocesamiento

class PostProcessor(tf.keras.layers.Layer):
    def __init__(self, num_classes, grid_size, anchors, anchor_mask):
        super(PostProcessor, self).__init__(trainable=False, dynamic=False, dtype='float32')
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.anchors = anchors
        self.anchor_mask = anchor_mask

    def call(self, inputs, training): # (B, G, G, A, 5 + N)
        batch_size = tf.shape(inputs)[0] # B
        xy, wh, p, c = tf.split(inputs, [2, 2, 1, self.num_classes], axis=-1) # (B, G, G, A, 2), (B, G, G, A, 2), (B, G, G, A, 1), (B, G, G, A, N)
        xy = tf.math.sigmoid(xy) # (B, G, G, A, 2)
        p = tf.math.sigmoid(p) # (B, G, G, A, 1)
        c = tf.math.softmax(c) # (B, G, G, A, N)
        # c = tf.math.sigmoid(c) # (B, G, G, A, N) # USE THIS ONLY WHEN C_LOSS IS BINARY CROSSENTROPY
        xywh = tf.concat([xy, wh], axis=-1) # (B, G, G, A, 4)
        grid = tf.meshgrid(tf.range(self.grid_size), tf.range(self.grid_size)) # [(G, G), (G, G)]
        grid = tf.stack(grid, axis=-1) # (G, G, 2)
        grid = tf.expand_dims(grid, axis=-2) # (G, G, 1, 2)
        grid = tf.expand_dims(grid, axis=0) # (1, G, G, 1, 2)
        grid = tf.tile(grid, [batch_size, 1, 1, len(self.anchor_mask), 1]) # (B, G, G, A, 2)
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

    def get_config(self):
        config = super(PostProcessor, self).get_config()
        config.update({'num_classes': self.num_classes, 'grid_size': self.grid_size, 'anchors': self.anchors, 'anchor_mask': self.anchor_mask})
        return config

#Capa nms
class NMS(tf.keras.layers.Layer):
    def __init__(self, num_classes, iou_threshold, score_threshold, max_bboxes):
        super(NMS, self).__init__(trainable=False, dynamic=False, dtype='float32')
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_bboxes = max_bboxes
        self.reshape_b13 = tf.keras.layers.Reshape(target_shape=(-1, 4))
        self.reshape_p13 = tf.keras.layers.Reshape(target_shape=(-1, 1))
        self.reshape_c13 = tf.keras.layers.Reshape(target_shape=(-1, self.num_classes))
        self.reshape_b26 = tf.keras.layers.Reshape(target_shape=(-1, 4))
        self.reshape_p26 = tf.keras.layers.Reshape(target_shape=(-1, 1))
        self.reshape_c26 = tf.keras.layers.Reshape(target_shape=(-1, self.num_classes))
        self.reshape_b52 = tf.keras.layers.Reshape(target_shape=(-1, 4))
        self.reshape_p52 = tf.keras.layers.Reshape(target_shape=(-1, 1))
        self.reshape_c52 = tf.keras.layers.Reshape(target_shape=(-1, self.num_classes))
        self.bbox_concat = tf.keras.layers.Concatenate(axis=1)
        self.p_concat = tf.keras.layers.Concatenate(axis=1)
        self.c_concat = tf.keras.layers.Concatenate(axis=1)
        self.reshape = tf.keras.layers.Reshape(target_shape=(-1, 1, 4))

    def call(self, inputs, training): # [[(B, 13, 13, 3, 4), (B, 13, 13, 3, 1), (B, 13, 13, 3, N)], [(B, 26, 26, 3, 4), (B, 26, 26, 3, 1), (B, 26, 26, 3, N)], [(B, 52, 52, 3, 4), (B, 52, 52, 3, 1), (B, 52, 52, 3, N)]]
        [bbox13, p13, c13], [bbox26, p26, c26], [bbox52, p52, c52] = inputs # [(B, 13, 13, 3, 4), (B, 13, 13, 3, 1), (B, 13, 13, 3, N)], [(B, 26, 26, 3, 4), (B, 26, 26, 3, 1), (B, 26, 26, 3, N)], [(B, 52, 52, 3, 4), (B, 52, 52, 3, 1), (B, 52, 52, 3, N)]
        bbox13 = self.reshape_b13(bbox13) # (B, 13 * 13 * 3, 4)
        p13 = self.reshape_p13(p13) # (B, 13 * 13 * 3, 1)
        c13 = self.reshape_c13(c13) # (B, 13 * 13 * 3, N)
        bbox26 = self.reshape_b26(bbox26) # (B, 26 * 26 * 3, 4)
        p26 = self.reshape_p26(p26) # (B, 26 * 26 * 3, 1)
        c26 = self.reshape_c26(c26) # (B, 26 * 26 * 3, N)
        bbox52 = self.reshape_b52(bbox52) # (B, 52 * 52 * 3, 4)
        p52 = self.reshape_p52(p52) # (B, 52 * 52 * 3, 1)
        c52 = self.reshape_c52(c52) # (B, 52 * 52 * 3, N)
        bbox = self.bbox_concat([bbox13, bbox26, bbox52]) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3), 4)
        p = self.p_concat([p13, p26, p52]) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3), 1)
        c = self.c_concat([c13, c26, c52]) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3), N)
        c = tf.argmax(c, axis=-1) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3))
        c = tf.cast(c, dtype=tf.float32) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3))
        c = tf.expand_dims(c, axis=-1) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3), 1)
        score = p * c # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3), 1)
        bbox = self.reshape(bbox) # (B, (13 * 13 * 3) + (26 * 26 * 3) + (52 * 52 * 3), 1, 4)
        bbox, score, c, valid = tf.image.combined_non_max_suppression(boxes=bbox, scores=score, max_output_size_per_class=self.max_bboxes, max_total_size=self.max_bboxes, iou_threshold=self.iou_threshold, score_threshold=self.score_threshold) # (B, M, 4), (B, M), (B, M), (B,)
        score = tf.expand_dims(score, axis=-1) # (B, M, 1)
        c = tf.expand_dims(c, axis=-1) # (B, M, 1)
        pred = tf.concat([bbox, score, c], axis=-1) # (B, M, 4 + 1 + 1)
        return pred, valid # (B, M, 4 + 1 + 1), (B,)

    def get_config(self):
        config = super(NMS, self).get_config()
        config.update({'num_classes': self.num_classes, 'iou_threshold': self.iou_threshold, 'score_threshold': self.score_threshold, 'max_bboxes': self.max_bboxes})
        return config


#DEFINICION DEL BLOQUE UPSAMPPLE
class upsamp(tf.keras.layers.Layer):
  def __init__(self,filters):
    super(upsamp,self).__init__()
    self.filters=filters
    self.up=tf.keras.layers.UpSampling2D(size=(2,2))
    self.conv=tf.keras.layers.Conv2D(self.filters/2.0,(3,3),(1,1),padding='same')
  
  def call(self,inputs):
    x=self.up(inputs)
    
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
  








  #Custom callback
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataframe, xcol, ycol, img_h, img_w, img_c, num_classes, grid_sizes, anchors, anchor_masks, max_bboxes):
        super(CustomCallback, self).__init__()
        self.dataframe = dataframe
        self.xcol = xcol
        self.ycol = ycol
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.num_classes = num_classes
        self.grid_sizes = grid_sizes
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.max_bboxes = max_bboxes

    def post_process(self, x, grid_size, anchor_mask):
        xy, wh, p, c = tf.split(x, [2, 2, 1, self.num_classes], axis=-1) # (1, G, G, A, 2), (1, G, G, A, 2), (1, G, G, A, 1), (1, G, G, A, N)
        xy = tf.math.sigmoid(xy) # (1, G, G, A, 2)
        p = tf.math.sigmoid(p) # (1, G, G, A, 1)
        # c = tf.math.sigmoid(c) # (1, G, G, A, N) # USE THIS ONLY WHEN C_LOSS IS BINARY CROSSENTROPY
        c = tf.math.softmax(c) # (1, G, G, A, N)
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size)) # [(G, G), (G, G)]
        grid = tf.stack(grid, axis=-1) # (G, G, 2)
        grid = tf.expand_dims(grid, axis=-2) # (G, G, 1, 2)
        grid = tf.expand_dims(grid, axis=0) # (1, G, G, 1, 2)
        grid = tf.tile(grid, [1, 1, 1, len(anchor_mask), 1]) # (1, G, G, A, 2)
        grid = tf.cast(grid, dtype=tf.float32) # (1, G, G, A, 2)
        xy = (xy + grid) / tf.cast(grid_size, dtype=tf.float32) # (1, G, G, A, 2)
        anch = tf.constant(self.anchors[anchor_mask], dtype=tf.float32) # (A, 2)
        anch = tf.expand_dims(anch, axis=0) # (1, A, 2)
        anch = tf.expand_dims(anch, axis=0) # (1, 1, A, 2)
        anch = tf.expand_dims(anch, axis=0) # (1, 1, 1, A, 2)
        anch = tf.tile(anch, [1, grid_size, grid_size, 1, 1]) # (1, G, G, A, 2)
        wh = tf.math.exp(wh) * anch # (1, G, G, A, 2)
        x1y1 = xy - (wh / 2.0) # (1, G, G, A, 2)
        x2y2 = xy + (wh / 2.0) # (1, G, G, A, 2)
        bbox = tf.concat([x1y1, x2y2], axis=-1) # (1, G, G, A, 2)
        bbox = tf.concat([bbox, p, c], axis=-1) # (1, G, G, 4 + 1 + N)
        bbox = tf.reshape(bbox, shape=(-1, 4 + 1 + self.num_classes)) # (1, G * G * A, 4 + 1 + N)
        return bbox

    def draw_bboxes(self, img_path, pred, grid_size, anchor_mask, rect_color, text_color):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(grid_size * grid_size * len(anchor_mask)):
            x1, y1, x2, y2, p, c = pred[i, 0:1], pred[i, 1:2], pred[i, 2:3], pred[i, 3:4], pred[i, 4:5], pred[i, 5:]

            c = np.argmax(c, axis=-1)
            x1 = int(x1 * self.img_w)
            y1 = int(y1 * self.img_h)
            x2 = int(x2 * self.img_w)
            y2 = int(y2 * self.img_h)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), rect_color, 2)
            img = cv2.putText(img, f'{c}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, text_color, 1)
        return img

    def on_epoch_end(self, epoch, logs=None):
        fig, ax = plt.subplots(1, 5, figsize=(25, 8))
        index = rn.randint(0, self.dataframe.shape[0] - 1)
        img_path = self.dataframe[self.xcol][index]
        bboxes = self.dataframe[self.ycol][index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for bbox in bboxes:
            x1, y1, x2, y2, cat, _ = bbox
            x1 = int(x1 * self.img_w)
            y1 = int(y1 * self.img_h)
            x2 = int(x2 * self.img_w)
            y2 = int(y2 * self.img_h)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img = cv2.putText(img, str(cat), (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        ax[0].imshow(img)
        ax[0].set_title('TRUE BBOXES')
        ax[0].axis('off')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = np.array(img, dtype='float32')
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        pred_all13, pred_all26, pred_all52 = self.model(img, True) # (B, 13, 13, 3, 5 + N), (B, 26, 26, 3, 5 + N), (B, 52, 52, 3, 5 + N)
        pred_all13 = self.post_process(pred_all13, self.grid_sizes[0], self.anchor_masks[0]).numpy() # (13 * 13 * A, 4 + 1 + N)
        img = self.draw_bboxes(img_path, pred_all13, self.grid_sizes[0], self.anchor_masks[0], (255, 0, 0), (0, 255, 255))
        ax[1].imshow(img)
        ax[1].set_title('13 x 13 GRID BBOXES')
        ax[1].axis('off')
        pred_all26 = self.post_process(pred_all26, self.grid_sizes[1], self.anchor_masks[1]).numpy() # (B, 26, 26, A, 4 + 1 + N)
        img = self.draw_bboxes(img_path, pred_all26, self.grid_sizes[1], self.anchor_masks[1], (0, 255, 0), (255, 0, 255))
        ax[2].imshow(img)
        ax[2].set_title('26 x 26 GRID BBOXES')
        ax[2].axis('off')
        pred_all52 = self.post_process(pred_all52, self.grid_sizes[2], self.anchor_masks[2]).numpy() # (B, 52, 52, A, 4 + 1 + N)
        img = self.draw_bboxes(img_path, pred_all52, self.grid_sizes[2], self.anchor_masks[2], (0, 0, 255), (255, 255, 0))
        ax[3].imshow(img)
        ax[3].set_title('52 x 52 GRID BBOXES')
        ax[3].axis('off')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = np.array(img, dtype='float32')
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        pred_best, valid_best = self.model(img) # (1, M, 4 + 1 + 1), (1,)
        pred_best = tf.squeeze(pred_best, axis=0).numpy() # (M, 4 + 1 + 1)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(self.max_bboxes):
            x1, y1, x2, y2, p, c = pred_best[i, 0:1], pred_best[i, 1:2], pred_best[i, 2:3], pred_best[i, 3:4], pred_best[i, 4:5], pred_best[i, 5:]
            x1 = int(x1 * self.img_w)
            y1 = int(y1 * self.img_h)
            x2 = int(x2 * self.img_w)
            y2 = int(y2 * self.img_h)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img = cv2.putText(img, f'{c}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
        ax[4].imshow(img)
        ax[4].set_title('NMS FINAL BBOXES')
        ax[4].axis('off')
        plt.show()


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, grid_size, anchors, anchor_mask, threshold):
        super(CustomLoss, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size # G
        self.anchors = anchors
        self.anchor_mask = anchor_mask
        self.grid = tf.meshgrid(tf.range(self.grid_size), tf.range(self.grid_size)) # [(G, G), (G, G)]
        self.grid = tf.stack(self.grid, axis=-1) # (G, G, 2)
        self.grid = tf.expand_dims(self.grid, axis=-2) # (G, G, 1, 2)
        self.grid = tf.cast(self.grid, dtype=tf.float32) # (G, G, 1, 2)
        self.anchor = self.anchors[self.anchor_mask] # (A, 2)
        self.anchor = tf.constant(self.anchor, dtype=tf.float32) # (A, 2)
        self.anchor = tf.expand_dims(self.anchor, axis=0) # (1, A, 2)
        self.anchor = tf.expand_dims(self.anchor, axis=0) # (1, 1, A, 2)
        self.anchor = tf.tile(self.anchor, [self.grid_size, self.grid_size, 1, 1]) # (G, G, A, 2)
        self.threshold = threshold

    def post_process(self, x): # (B, G, G, A, 5 + N)
        batch_size = tf.shape(x)[0] # B
        xy, wh, p, c = tf.split(x, [2, 2, 1, self.num_classes], axis=-1) # (B, G, G, A, 2), (B, G, G, A, 2), (B, G, G, A, 1), (B, G, G, A, N)
        xy = tf.math.sigmoid(xy) # (B, G, G, A, 2)
        p = tf.math.sigmoid(p) # (B, G, G, A, 1)
        c = tf.math.softmax(c) # (B, G, G, A, N)
        # c = tf.math.sigmoid(c) # (B, G, G, A, N) # USE THIS ONLY WHEN C_LOSS IS BINARY CROSSENTROPY
        xywh = tf.concat([xy, wh], axis=-1) # (B, G, G, A, 4)
        grid = tf.meshgrid(tf.range(self.grid_size), tf.range(self.grid_size)) # [(G, G), (G, G)]
        grid = tf.stack(grid, axis=-1) # (G, G, 2)
        grid = tf.expand_dims(grid, axis=-2) # (G, G, 1, 2)
        grid = tf.expand_dims(grid, axis=0) # (1, G, G, 1, 2)
        grid = tf.tile(grid, [batch_size, 1, 1, len(self.anchor_mask), 1]) # (B, G, G, A, 2)
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
        bbox = tf.concat([x1y1, x2y2], axis=-1) # (B, G, G, A, 2)
        return bbox, p, c, xywh

    def broadcast_dynamic_shape_and_get_iou_per_image(self, bbox_pred, bbox_true): # bbox_pred: (G, G, A, 4), bbox_true: (R, 4)
        bbox_pred = tf.expand_dims(bbox_pred, axis=-2) # (G, G, A, 1, 4)
        bbox_true = tf.expand_dims(bbox_true, axis=0) # (1, R, 4)
        new_shape = tf.broadcast_dynamic_shape(tf.shape(bbox_pred), tf.shape(bbox_true)) # [G, G, A, R, 4]
        bbox_pred = tf.broadcast_to(bbox_pred, new_shape) # (G, G, A, R, 4)
        bbox_true = tf.broadcast_to(bbox_true, new_shape) # (G, G, A, R, 4)
        x2_int = tf.minimum(bbox_pred[..., 2], bbox_true[..., 2]) # (G, G, A, R)
        x1_int = tf.maximum(bbox_pred[..., 0], bbox_true[..., 0]) # (G, G, A, R)
        y2_int = tf.minimum(bbox_pred[..., 3], bbox_true[..., 3]) # (G, G, A, R)
        y1_int = tf.maximum(bbox_pred[..., 1], bbox_true[..., 1]) # (G, G, A, R)
        w_int = tf.maximum(x2_int - x1_int, 0.0) # (G, G, A, R)
        h_int = tf.maximum(y2_int - y1_int, 0.0) # (G, G, A, R)
        area_int = w_int * h_int # (G, G, A, R)
        area_true = (bbox_true[..., 2] - bbox_true[..., 0]) * (bbox_true[..., 3] - bbox_true[..., 1]) # (G, G, A, R)
        area_pred = (bbox_pred[..., 2] - bbox_pred[..., 0]) * (bbox_pred[..., 3] - bbox_pred[..., 1]) # (G, G, A, R)
        iou = area_int / (area_true + area_pred - area_int) # (G, G, A, R)
        return iou

    def get_best_iou(self, bbox_pred, bbox_true, obj_mask): # (G, G, A, 4), (G, G, A, 4), (G, G, A)
        obj_mask = tf.cast(obj_mask, dtype=tf.bool) # (G, G, A)
        bbox_true = tf.boolean_mask(bbox_true, obj_mask) # (R, 4)
        iou = self.broadcast_dynamic_shape_and_get_iou_per_image(bbox_pred, bbox_true) # (G, G, A, R)
        best_iou = tf.reduce_max(iou, axis=-1) # (G, G, A)
        return best_iou

    def call(self, y_true, y_pred): # y_true: [(B, G, G, A, 4), (B, G, G, A, 1), (B, G, G, A, 1), (B, G, G, A, 4)], y_pred: (B, G, G, A, 5 + N)
        bbox_pred, p_pred, c_pred, xywh_pred = self.post_process(y_pred) # (B, G, G, A, 4), (B, G, G, A, 1), (B, G, G, A, N), (B, G, G, A, 4)
        xy_pred, wh_pred = tf.split(xywh_pred, [2, 2], axis=-1) # (B, G, G, A, 2), (B, G, G, A, 2)

        bbox_true, p_true, c_true, xywh_true = tf.split(y_true, [4, 1, 1, 4], axis=-1) # (B, G, G, A, 4), (B, G, G, A, 1), (B, G, G, A, 1), (B, G, G, A, 4)
        xy_true, wh_true = tf.split(xywh_true, [2, 2], axis=-1) # (B, G, G, A, 2), (B, G, G, A, 2)

        bbox_scale = 2.0 - (wh_true[..., 0] * wh_true[..., 1]) # (B, G, G, A, 1)

        xy_true = (xy_true * tf.cast(self.grid_size, dtype=tf.float32)) - self.grid # (B, G, G, A, 2)

        wh_true = tf.math.log(wh_true / self.anchor) # (B, G, G, A, 2)
        wh_true = tf.where(tf.math.is_inf(wh_true), tf.zeros_like(wh_true), wh_true) # (B, G, G, A, 2)

        obj_mask = tf.squeeze(p_true, axis=-1) # (B, G, G, A)
        best_iou = tf.map_fn(lambda x: self.get_best_iou(x[0], x[1], x[2]), [bbox_pred, bbox_true, obj_mask], fn_output_signature=tf.float32) # (B, G, G, A)
        ignore_mask = tf.cast(best_iou < self.threshold, dtype=tf.float32) # (B, G, G, A)

        xy_loss = obj_mask * bbox_scale * tf.reduce_sum(tf.math.square(xy_true - xy_pred), axis=-1) # (B, G, G, A)
        wh_loss = obj_mask * bbox_scale * tf.reduce_sum(tf.math.square(wh_true - wh_pred), axis=-1) # (B, G, G, A)
        c_loss = obj_mask * tf.keras.losses.sparse_categorical_crossentropy(c_true, c_pred) # (B, G, G, A)

        alpha = 0.85
        p_loss = tf.keras.losses.binary_crossentropy(p_true, p_pred) # (B, G, G, A)
        p_loss = tf.math.pow(obj_mask - tf.squeeze(tf.math.sigmoid(p_pred), axis=-1), 2) * ((1.0 - alpha) * obj_mask * p_loss + alpha * (1.0 - obj_mask) * ignore_mask * p_loss) # (B, G, G, A)

        xy_loss = tf.reduce_sum(xy_loss, axis=[1, 2, 3]) # (B,)
        wh_loss = tf.reduce_sum(wh_loss, axis=[1, 2, 3]) # (B,)
        p_loss = tf.reduce_sum(p_loss, axis=[1, 2, 3]) # (B,)
        c_loss = tf.reduce_sum(c_loss, axis=[1, 2, 3]) # (B,)

        loss = xy_loss + wh_loss + p_loss + c_loss # (B,)
        return loss
