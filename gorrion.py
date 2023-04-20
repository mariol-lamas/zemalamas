#----------------------------------------------------# IMPORTACION DE PAQUETES #------------------------------------------------------#
import os

import customtkinter
import time

import numpy as np
from ultralytics import YOLO
import cv2
import supervision as sv
import tkinter as tk

import customtkinter as ctk
from PIL import Image, ImageTk
import imutils

import cv2



from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1


from scipy.spatial.distance import cosine
import face_recognition
from funciones_Programa import *
customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")



#------------------------------------------------------# CREACION DEL UI #------------------------------------------------------#
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.ruta_prog = os.getcwd()
        self.title("Multidector by m.l.")
        self.geometry(f"{1100}x{580}")
        self.resizable(width=False,height=False)

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="MENU",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame,text='INICIO',command=self.frame_inicio)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame,text='RECONOCIMIENTO\nFACIAL', command=self.frame_recon)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame,text='DETECTOR',command=self.frame_detect)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text='SALIR', command=self.salir)
        self.sidebar_button_4.grid(row=4, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appariencia:", anchor="w")
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Claro", "Oscuro"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=7, column=0, padx=20, pady=(10, 10))
        #Valores por defecto
        customtkinter.set_appearance_mode("Light")
        self.frame_inicio()
        self.modo='Webcam'
        self.ruta=''
        comprobar_ruta(self.ruta_prog)

        self.imagenes_dir = len(os.listdir(f'{self.ruta_prog}/Imagenes'))

        self.caras = False
        self.guardar = False

    ##################################################################################################################################################################
    # ------------------------------------------- DEFINICION  DEL FRAME DE INICIO --------------------------------------#
    ##################################################################################################################################################################

    def frame_inicio(self):
        frame_ini = ctk.CTkFrame(self,width=650,height=550,corner_radius=10)
        frame_ini.grid(row=0,column=1,pady=12,sticky='nsew')
        frame_ini.grid_columnconfigure((2, 3), weight=0)
        frame_ini.grid_rowconfigure((0, 1, 2), weight=1)
        titulo_ini=ctk.CTkLabel(frame_ini,text='BIENVENIDO A EL MULTIDETECTOR',font=ctk.CTkFont(size=28,weight='bold'))
        titulo_ini.place(x=80,y=30)
        textbox = customtkinter.CTkTextbox(frame_ini, width=600,height=400)
        textbox.place(x=30,y=80)
        textbox.insert('0.0',text='')
        frame_ini.tkraise()
##################################################################################################################################################################
# ------------------------------------------- DEFINICION  DEL FRAME DE RECONOCIMIENTO --------------------------------------#
##################################################################################################################################################################

    def frame_recon(self):
        ###Detector MCNN###
        # --------------------#
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        mtcnn = MTCNN(
            select_largest=True,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False,
            image_size=160,
            device=device
        )
        ### ENCODER ###
        # ----------------#
        encoder = InceptionResnetV1(
            pretrained='vggface2',
            classify=False,
            device=device
        ).eval()
        ##########################################################################################
        ###############         DEFINICION DE LAS FUNCIONES DE RECON                ###############
        ##########################################################################################
        def recon_nuev_im():
            if len(f'{self.ruta_prog}/Imagenes')>0:
                for elem in os.listdir(f'{self.ruta_prog}/Imagenes'):
                    im =cv2.imread(f'{self.ruta_prog}/Imagenes/{elem}')
                    localizacion_caras = face_recognition.face_locations(im)
                    print(localizacion_caras)
                    if localizacion_caras is not None:
                        for cara in localizacion_caras:
                            cara = cara[cara[0]:cara[2],cara[3]:cara[1]]
                            encoding = calcular_embedding(cara,encoder)
                            comprobar_similitud(encoding)

        def comprobar_similitud(encoding):
            if len(os.listdir(f'{self.ruta_prog}/Base_Datos_Personas'))>0:
                for pers in os.listdir(f'{self.ruta_prog}/Base_Datos_Personas'):
                    encoding_comp = np.loadtxt(f'{self.ruta_prog}/Base_Datos_Personas/{pers}/encoding.csv',delimiter=',')
                    if (1-cosine(encoding_comp,encoding))>0.6:
                        encoding_nuev = np.mean(encoding_comp,encoding)
                        np.savetxt(f'{self.ruta_prog}/Base_Datos_Personas/{pers}/encoding.csv',encoding_nuev,delimiter=',')
            #else:









        ##########################################################################################
        #                        DEFINICION DEL FRAME DE RECONOCIMIENTO
        ##########################################################################################

        self.frame_rec = ctk.CTkFrame(self,width=650,height=550,corner_radius=10)
        self.frame_rec.grid(row=0,column=1,pady=12,sticky='nsew')
        textbox = customtkinter.CTkTextbox(self.frame_rec, width=250, height=150)
        textbox.place(x=665, y=310)
        textbox.insert('0.0', text=(f'Modo de deteccion: {self.modo}\n'
                                    f''))
        boton = ctk.CTkButton(self.frame_rec, text='Iniciar Reconocimiento')
        boton.place(x=150, y=450)
        boton1 = ctk.CTkButton(self.frame_rec, text='Parar Reconocimiento')
        boton1.place(x=350, y=450)
        self.titulo_recon = customtkinter.CTkLabel(self.frame_rec, text="RECONOCIMIENTO FACIAL",
                                                    font=customtkinter.CTkFont(size=26, weight="bold"))
        self.titulo_recon.place(x=150, y=10)
        entrada_texto = ctk.CTkEntry(self.frame_rec, placeholder_text="Introduce un commando", width=480)
        entrada_texto.place(x=10, y=500)
        boton_entrada_texto = ctk.CTkButton(self.frame_rec, text='Enviar')
        boton_entrada_texto.place(x=510, y=500)

        lateral_frame = customtkinter.CTkFrame(self.frame_rec, width=230,height=270)
        lateral_frame.place(x=670, y=10)

        boton_1_der = ctk.CTkButton(lateral_frame, text='IMAGENES POR DETECTAR', command=recon_nuev_im)
        boton_1_der.place(x=50, y=60)
        boton_2_der = ctk.CTkButton(lateral_frame, text='Enviar2')
        boton_2_der.place(x=50, y=110)
        boton_3_der = ctk.CTkButton(lateral_frame, text='Enviar3')
        boton_3_der.place(x=50, y=160)
        boton_4_der = ctk.CTkButton(lateral_frame, text='Enviar4')
        boton_4_der.place(x=50, y=210)
        self.etiq_video_recon = tk.Label(self.frame_rec)
        self.etiq_video_recon.place(x=10, y=50)
        self.frame_rec.tkraise()




##################################################################################################################################################################
# ------------------------------------------- DEFINICION  DEL FRAME DEL DETECTOR --------------------------------------#
##################################################################################################################################################################


    def frame_detect(self):
        self.video = None
        self.ret = False
        modelo = YOLO('yolov8n.pt')
        anota_cubos = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )
        self.lista_labels=['persona', 'bicicleta', 'coche', 'motocicleta', 'avion', 'autobus', 'tren', 'camion', 'barco', 'semaforo',
         'hidrante', 'señal de stop', 'parquimetro', 'banco', 'pajaro', 'gato', 'perro', 'caballo', 'oveja', 'vaca',
         'elefante', 'oso', 'cebra', 'jirafa', 'mochila', 'paraguas', 'bolso', 'corbata', 'maleta', 'frisbee', 'esquis',
         'tabla de snowboard', 'pelota de deportes', 'cometa', 'bate de beisbol', 'guante de beisbol', 'patineta',
         'tabla de surf', 'raqueta de tenis', 'botella', 'copa de vino', 'taza', 'tenedor', 'cuchillo', 'cuchara',
         'cuenco', 'platano', 'manzana', 'sandwich', 'naranja', 'brocoli', 'zanahoria', 'perro caliente', 'pizza',
         'dona', 'pastel', 'silla', 'sofa', 'planta en maceta', 'cama', 'mesa de comedor', 'inodoro', 'television',
         'portatil', 'raton', 'control remoto', 'teclado', 'telefono celular', 'microondas', 'horno', 'tostadora',
         'fregadero', 'refrigerador', 'libro', 'reloj', 'jarron', 'tijeras', 'osito de peluche', 'secador de pelo',
         'cepillo de dientes']


#------------------------------------------- DEFINICION DE LAS FUNCIONES DEL FRAME DEL DETECTOR --------------------------------------#

        def deteccion(clase):
            inicio =time.time()
            if self.video == None:
                ...
            else:
                self.ret, frame = self.video.read()
                if self.ret == True and len(clase)>0 and self.checkbox_1.get()==1:
                    resultaados = modelo(frame, conf=0.5, classes=clase, verbose=False)[0]
                    detecciones = sv.Detections.from_yolov8(resultaados)
                    try:
                        array = [
                            array
                            for array, _, class_od, _ in detecciones if class_od == 0
                        ][0]
                        if self.caras == True and array is not None:
                            persona = frame[round(array[1]):round(array[3]), round(array[0]):round(array[2])]
                            cv2.imwrite(f'{self.ruta_prog}/Imagenes/{self.imagenes_dir}.jpg', persona)
                            self.imagenes_dir += 1
                    except:
                        IndexError

                    nombres = [
                        f'{self.lista_labels[class_id]}'  # {confidence:0.2f}'
                        for _, confidence, class_id, _
                        in detecciones
                    ]
                    frame = anota_cubos.annotate(scene=frame, detections=detecciones, labels=nombres)
                    frame = imutils.resize(frame, width=640, height=480)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    img = ImageTk.PhotoImage(image=img)
                    etiq_video.configure(image=img)
                    etiq_video.image = img
                    etiq_video.after(2, main)
                else:
                    detener()
                    print('\nPara iniciar la deteccion debes elegir las clases y pulsar seleccion preparada\n')
            fin = time.time()
            print('TIEMPO: ', fin-inicio)
        def main():
            etiq_video.place(x=10, y=60)

            if self.modo=='Video' and self.checkbox_1.get()==1:
                clase = clases_para_detect()
                deteccion(clase)
            elif self.modo=='Webcam'and self.checkbox_1.get()==1:
                clase = clases_para_detect()
                deteccion(clase)
            elif self.modo=='Imagen'and self.checkbox_1.get()==1:
                clase = clases_para_detect()
                img=cv2.imread(self.ruta)
                result=modelo(img,conf=0.5, classes=clase,verbose=False)[0]
                detecciones = sv.Detections.from_yolov8(result)
                nombres = [
                    f'{self.lista_labels[class_id]}'  # {confidence:0.2f}'
                    for _, confidence, class_id, _
                    in detecciones
                ]
                frame = anota_cubos.annotate(scene=img, detections=detecciones, labels=nombres)
                frame = imutils.resize(frame, width=640, height=480)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img=img.resize((640,380),Image.ANTIALIAS)
                img = ImageTk.PhotoImage(image=img)
                print(img)
                etiq_video.image = img
                etiq_video.configure(image=img)
                etiq_video.image = img

            else:
                print('\nEl modo de deteccion no es el correcto o el boton seleccion preparada no esta clicado\n')
        # clases_para_detect se encarga de comprobar os switches que estan activos es decir las clases seleccionadas
        # para la deteccion
        def clases_para_detect():
            self.list=[]
            dic = {'persona': 0, 'bicicleta': 1, 'coche': 2, 'motocicleta': 3, 'avion': 4, 'autobus': 5, 'tren': 6, 'camion': 7, 'barco': 8, 'semaforo': 9, 'hidrante': 10, 'señal de stop': 11, 'parquimetro': 12, 'banco': 13, 'pajaro': 14, 'gato': 15, 'perro': 16, 'caballo': 17, 'oveja': 18, 'vaca': 19, 'elefante': 20, 'oso': 21, 'cebra': 22, 'jirafa': 23, 'mochila': 24, 'paraguas': 25, 'bolso': 26, 'corbata': 27, 'maleta': 28, 'frisbee': 29, 'esquis': 30,
                   'tabla de snowboard': 31, 'pelota de deportes': 32, 'cometa': 33, 'bate de beisbol': 34, 'guante de beisbol': 35, 'patineta': 36, 'tabla de surf': 37, 'raqueta de tenis': 38, 'botella': 39, 'copa de vino': 40, 'taza': 41, 'tenedor': 42, 'cuchillo': 43, 'cuchara': 44, 'cuenco': 45, 'platano': 46, 'manzana': 47, 'sandwich': 48, 'naranja': 49, 'brocoli': 50, 'zanahoria': 51, 'perro caliente': 52, 'pizza': 53, 'dona': 54, 'pastel': 55, 'silla': 56,
                   'sofa': 57, 'planta en maceta': 58, 'cama': 59, 'mesa de comedor': 60, 'inodoro': 61, 'television': 62, 'portatil': 63, 'raton': 64, 'control remoto': 65, 'teclado': 66, 'telefono celular': 67, 'microondas': 68, 'horno': 69, 'tostadora': 70, 'fregadero': 71, 'refrigerador': 72, 'libro': 73, 'reloj': 74, 'jarron': 75, 'tijeras': 76, 'osito de peluche': 77, 'secador de pelo': 78, 'cepillo de dientes': 79}

            lista=self.comprobar_switches()
            if len(lista)==80:
                self.list=lista
            else:
                for eleme in lista:
                    self.list.append(dic[eleme])
            return self.list
        # iniciar_web enciende la webcam y llama a la funcion de deteccion por webcam
        def iniciar_web():

            if self.modo=='Webcam':
                self.video = cv2.VideoCapture(0)
            elif self.modo=='Video':
                self.video = cv2.VideoCapture(self.ruta)
            elif self.modo=='Imagen':
                ...
            main()
        # detener, detiene la ejecucion de la deteccion
        def detener():
            self.video = None
            etiq_video.place_forget()
            etiq_video.image = ''
            self.ret = False
        # soltar_listo, al seleccionar un nuevo switch teniendo marcada la opcion de seleccion lista, la deselecciona
        # esto evita errores durante la ejecucion
        def soltar_listo():
            if self.checkbox_1.get()==1:
                self.checkbox_1.deselect()
                detener()
            else:
                ...
        #-------------------------------------------#
        #DEFINICION DE LOS COMANDO QUE SE PUEDEN EJECUTAR EN LA VENTANA DE DETECCION
        #-------------------------------------------#
        def comprobacion_video(ruta):
            valor = ruta.rsplit(sep='.',maxsplit=1)
            if valor[1].lower()==('mp4'or'mov'):
                ...
            else:
                print('\n El formato de video introducido no es válido\n'
                      'Los formatos permitidos son .mp4 .mov\n ')
        def comprobacion_img(ruta):
            valor = ruta.rsplit(sep='.', maxsplit=1)
            print(ruta)
            if valor[1] == ('jpg' or 'png'or 'jpeg'):
                ...
            else:
                print('\n El formato de la imagen introducido no es válido\n '
                      'Los formatos permitidos son .jpg .png .jpeg\n')




        def comandos():
            text_lst=entrada_texto.get().split(sep=' ',maxsplit=1)

            if len(text_lst)==0:
                print('No se ha introducido ningun comando')
            else:
                texto = text_lst[0]
                if len(text_lst)==2 and texto!='Buscar':
                    ruta = text_lst[1]

                    if os.path.exists(ruta)==False:
                        print('La ruta introducida no existe, introduce la ruta absoluta')
                    else:
                        if texto=='imagen':
                            self.modo='Imagen'
                            self.ruta = ruta
                            comprobacion_img(self.ruta)
                            textbox.delete('1.0','end')
                            textbox.insert('0.0', text=(f'Modo de deteccion: {self.modo}\n'
                                                f''))

                        elif texto=='video':
                            self.modo = 'Video'
                            self.ruta=ruta
                            textbox.delete('1.0', 'end')
                            textbox.insert('0.0', text=(f'Modo de deteccion: {self.modo}\n'
                                                f''))
                            comprobacion_video(self.ruta)
                        else:
                            print('\nEl comando introducido no existe revisa el manual de comandos\n')
                elif len(text_lst)==2 and texto=='Buscar':
                    archivo = text_lst[1]
                    buscar_fichero(archivo)

                else:
                    if texto=='webcam':
                        self.modo = 'Webcam'
                        textbox.delete('1.0', 'end')
                        textbox.insert('0.0', text=(f'Modo de deteccion: {self.modo}\n'
                                                f''))
                    else:
                        print('\n El comando introducido no existe\n')

        ##########################################################################################
                    #Definicion del resto de widgets del frame detector
        ##########################################################################################

        frame_dete = ctk.CTkFrame(self,width=650,height=550,corner_radius=10)
        frame_dete.grid(row=0,column=1,pady=12,sticky='nsew')
        etiq_video = tk.Label(frame_dete)
        etiq_video.place(x=10, y=50)
        boton = ctk.CTkButton(frame_dete, text='Iniciar Deteccion', command=iniciar_web)
        boton.place(x=150, y=450)
        boton1 = ctk.CTkButton(frame_dete, text='Parar Deteccion', command=detener)
        boton1.place(x=350, y=450)
        scrollable_frame = customtkinter.CTkScrollableFrame(frame_dete, label_text="Parametros para deteccion",label_font=ctk.CTkFont(size=16,weight='bold'))
        scrollable_frame.place(x=670,y=10)
        scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame_switches = []
        self.lista_objetos=['Todos','persona', 'bicicleta', 'coche', 'motocicleta', 'avion', 'autobus', 'tren', 'camion', 'barco', 'semaforo',
         'hidrante', 'señal de stop', 'parquimetro', 'banco', 'pajaro', 'gato', 'perro', 'caballo', 'oveja', 'vaca',
         'elefante', 'oso', 'cebra', 'jirafa', 'mochila', 'paraguas', 'bolso', 'corbata', 'maleta', 'frisbee', 'esquis',
         'tabla de snowboard', 'pelota de deportes', 'cometa', 'bate de beisbol', 'guante de beisbol', 'patineta',
         'tabla de surf', 'raqueta de tenis', 'botella', 'copa de vino', 'taza', 'tenedor', 'cuchillo', 'cuchara',
         'cuenco', 'platano', 'manzana', 'sandwich', 'naranja', 'brocoli', 'zanahoria', 'perro caliente', 'pizza',
         'dona', 'pastel', 'silla', 'sofa', 'planta en maceta', 'cama', 'mesa de comedor', 'inodoro', 'television',
         'portatil', 'raton', 'control remoto', 'teclado', 'telefono celular', 'microondas', 'horno', 'tostadora',
         'fregadero', 'refrigerador', 'libro', 'reloj', 'jarron', 'tijeras', 'osito de peluche', 'secador de pelo',
         'cepillo de dientes']

        i=0
        for elem in self.lista_objetos:
            self.switch = ctk.CTkSwitch(master=scrollable_frame, text=f"{elem}",command=soltar_listo)
            self.switch.grid(row=i, column=0, padx=10, pady=(0, 20))
            i+=1
            self.scrollable_frame_switches.append(self.switch)
        # Este bucle crea los switches con los nombres de la lista objetos
        self.checkbox_1 = customtkinter.CTkCheckBox(frame_dete,text='Seleccion preparada',command=self.comprobar_switches)
        self.checkbox_1.place(x=670,y=275)
        entrada_texto = ctk.CTkEntry(frame_dete, placeholder_text="Introduce un commando",width=480)
        entrada_texto.place(x=10,y=500)
        boton_entrada_texto =ctk.CTkButton(frame_dete,text='Enviar',command=comandos)
        boton_entrada_texto.place(x=510,y=500)
        textbox = customtkinter.CTkTextbox(frame_dete, width=250,height=150)
        textbox.place(x=665, y=310)
        textbox.insert('0.0', text=(f'Modo de deteccion: {self.modo}\n'
                                    f''))
        self.switch_guardar = ctk.CTkSwitch(frame_dete, text='Guardar output')
        self.switch_guardar.place(x=665, y= 470 )
        self.switch_caras = ctk.CTkSwitch(frame_dete, text='Extraer personas')
        self.switch_caras.place(x=665, y=500)
        self.titulo_detect = customtkinter.CTkLabel(frame_dete, text="DETECTOR",
                                                 font=customtkinter.CTkFont(size=26, weight="bold"))
        self.titulo_detect.place(x=250,y=10)
        frame_dete.tkraise()

##################################################################################################################################################################
# ------------------------------------------- DEFINICION  DE OTRAS FUNCIONES GENERALES --------------------------------------#
##################################################################################################################################################################

    def mostar_frame(self,frame_llamado):
        frame=self.todos_frames[frame_llamado]
        frame.tkraise()

    def change_appearance_mode_event(self, new_appearance_mode: str):
        dic={'Claro':'Light','Oscuro':'Dark'}
        valor = dic[new_appearance_mode]
        ctk.set_appearance_mode(valor)

    def comprobar_switches(self):
        lista=[]
        self.lista_objetos
        if self.switch_caras.get()==1:
            self.caras=True
        else:
            self.caras = False
        if self.switch_guardar.get()==1:
            self.guardar=True
        else:
            self.guardar = False

        i=0

        for elem in self.lista_objetos:
            if self.scrollable_frame_switches[i].get()==1 and self.scrollable_frame_switches[i]._text=='Todos':
                lista = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                         50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
                         74, 75, 76, 77, 78, 79]

            elif self.scrollable_frame_switches[i].get()==1 and self.scrollable_frame_switches[i]._text!='Todos':
                lista.append(elem)
            i+=1
        return lista
    def salir(self):
        app.destroy()


##################################################################################################################################################################
# ------------------------------------------- FIN DEL PROGRAMA --------------------------------------#
##################################################################################################################################################################
class vent_nuev_user(ctk.CTk):
    def __init__(self,parent):
        super().__init__()

        boton_cerar=ctk.CTkButton(self,text='Cerrar',command=self.cerrar)
        boton_cerar.pack()
        self.mainloop()

    def cerrar(self):
        self.destroy()



if __name__ == "__main__":
    app = App()
    app.mainloop()

