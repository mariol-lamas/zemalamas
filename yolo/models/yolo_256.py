##############################################3
        #IMPORTACION DE PAQUETES
##############################################
import tensorflow as tf
from models.utils.layer_types import CONV, C2f, Output, PostProcessor, NMS,CustomCallback, SPPF, upsamp



#DEFINICION DE LOS DISTINTOS MODELOS

class YOLO(tf.keras.Model):
    def __init__(self, num_classes, grid_sizes, anchors, anchor_masks, iou_threshold, score_threshold, max_bboxes):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        self.grid_sizes = grid_sizes
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_bboxes = max_bboxes

        #Capas para el modelo
        self.conv1=CONV((3,3),(2,2),64) #Tam salida  (1,320,320,64)
        self.conv2=CONV((3,3),(2,2),128) #Tam salida (1,160,160,128)
        self.c2f_1=C2f(128,(3,3),(1,1),True,1) #Tam salida (1,160,160,128)
        self.conv3=CONV((3,3),(2,2),256) #Tam salida (1,80,80,256)
        self.c2f_2=C2f(256,(3,3),(1,1),True,1) #Tam salida (1,80,80,256)
        self.conv4=CONV((3,3),(2,2),256) #Tam salida (1,40,40,256)
        self.c2f_3=C2f(256,(3,3),(1,1),True,1) #Tam salida (1,40,40,256)
        self.conv5=CONV((3,3),(2,2),256) #Tam salida (1,20,20,256)
        self.c2f_4=C2f(256,(3,3),(1,1),True,1) #Tam salida (1,20,20,256)      
        self.c2f_5=C2f(256,(3,3),(1,1),False,1) #Tam salida (1,40,40,512)
        self.c2f_6=C2f(256,(2,2),(1,1),False,1) #Tam salida (1,80,80,51)
        self.conv6=CONV((3,3),(2,2),256) #Tam salida (1,40,40,256)
        self.c2f_7=C2f(256,(3,3),(1,1),False,1) #Tam salida (1,40,40,512)
        self.conv7=CONV((3,3),(2,2),256) #Tam salida (1,20,20,256)
        self.c2f_8=C2f(256,(3,3),(1,1),False,1) #Tam salida (1,20,20,512)
        self.conv8=CONV((3,3),(1,1),256) #Tam salida (1,20,20,256)

        self.sppf=SPPF(256)
        self.up_20=upsamp(256)
        self.up_40=upsamp(256)
        self.concat=tf.keras.layers.Concatenate(axis=-1)

        #Layers de salida
        self.out80=Output(256,self.num_classes,self.grid_sizes[2],self.anchors[self.anchor_masks[2]])
        self.out40=Output(256,self.num_classes,self.grid_sizes[1],self.anchors[self.anchor_masks[1]])
        self.out20=Output(256,self.num_classes,self.grid_sizes[0],self.anchors[self.anchor_masks[0]])

        #Postprocesamiento
        self.post1 = PostProcessor(self.num_classes, self.grid_sizes[0], self.anchors, self.anchor_masks[0])
        self.post2 = PostProcessor(self.num_classes, self.grid_sizes[1], self.anchors, self.anchor_masks[1])
        self.post3 = PostProcessor(self.num_classes, self.grid_sizes[2], self.anchors, self.anchor_masks[2])

        #Capa nms para preparar preds
        self.nms = NMS(self.num_classes, self.iou_threshold, self.score_threshold, self.max_bboxes)


    
    def call(self, inputs, training=False):
        x=self.conv1(inputs)
        x=self.conv2(x)
        x=self.c2f_1(x)
        x=self.conv3(x)
        x80=self.c2f_2(x) #salida 80x80 (1,80,80,256)


        x=self.conv4(x80)
        x40=self.c2f_3(x) #salida 40x40 (1,40,40,256)


        x=self.conv5(x40)
        x=self.c2f_4(x)
        x20=self.sppf(x) #salida 20x20 (1,20,20,256)

        x=self.up_20(x20) #sale (1,40,40,256)
        print(x.shape)
        x=tf.add(x,x40)/2.0 #(1,40,40,256
        print(x.shape)
        x=self.c2f_5(x)

        #x1,x2=tf.split(x,2,axis=-1) # (1,40,40,256) | (1,40,40,256)
        x1=self.up_40(x) #(1,80,80,256)
        x1=tf.add(x1,x80)/2.0 #(1,80,80,256)
        x80=self.c2f_6(x1)
        #x80,x1=tf.split(x1,2,axis=-1) #(1,80,80,256) SALIDA 80x80| (1,80,80,256)
        x1=self.conv6(x80)

        x2=tf.add(x1,x)/2.0 #sal (1,40,40,256)
        x40=self.c2f_7(x2) # (1,40,40,256)
        #x2,x40=tf.split(x2,2,axis=-1) # (1,40,40,256) | (1,40,40,256) SALIDA 40x40
        x2=self.conv7(x40) #(1,20,20,256)
        x2=tf.add(x2,x20)/2.0
        x2=self.c2f_8(x2)
        x20=self.conv8(x2) #(1,20,20,256) SALIDA 20x20


        x80_out=self.out80(x80)
        x40_out=self.out40(x40)
        x20_out=self.out20(x20)

        if training:
            return [x20_out,x40_out,x80_out]
        
        bbox20, p20, c20, _ = self.post1(x20_out, False)
        bbox40, p40, c40, _ = self.post2(x40_out, False)
        bbox80, p80, c80, _ = self.post3(x80_out, False)
        pred, valid = self.nms([[bbox20, p20, c20], [bbox40, p40, c40], [bbox80, p80, c80]], False) 

        return pred, valid

    def summary(self, img_h, img_w, img_c, batch_size):
        X_inp = tf.keras.Input(shape=(img_h, img_w, img_c), batch_size=batch_size)
        x = self.call(X_inp)
        m = tf.keras.Model(inputs=X_inp, outputs=x)
        return m.summary()


