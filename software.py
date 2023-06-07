#----------------------------------------------------# IMPORTACION DE PAQUETES #------------------------------------------------------#
import os,signal
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
import psycopg2
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib
import cv2
import subprocess
import pyautogui
import platform


#Valores por defecto para la apariencia de las ventanas
customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")

#DEFINICION DEL FRAME PARA LA TABLA
class Table(tk.Frame):
    def __init__(self, parent=None, title='',headers=None,height=10,*args,**kwargs):
        tk.Frame.__init__(self, parent,*args,**kwargs)
        self._title=tk.Label(self, text=title)
        self._headers=headers
        self._tree=tk.ttk.Treeview(self, height=height,columns=self._headers, show='headings')
        self._title.pack(side=tk.TOP, fill='x')

        #Agregamos los scrollbacks laterales
        vert=tk.ttk.Scrollbar(self, orient='vertical', command=self._tree.yview)
        vert.pack(side='right',fill='y')
        hor=tk.ttk.Scrollbar(self, orient='horizontal', command=self._tree.xview)
        hor.pack(side='bottom',fill='x')

        #Acabamos de configurar la tabla
        self._tree.configure(xscrollcommand=hor.set, yscrollcommand=vert.set)
        self._tree.pack(side='left')

        #Introducimos el titulo de las columnas
        for tit in self._headers:
            self._tree.heading(tit, text=tit.title())
            self._tree.column(tit, stretch=True, width=180)
    
    #----------
    #Funcion encargada de insertar la informacion de una fila en la tabla
    #-----------
    def anadir_fila(self, fila):
        self._tree.insert('','end',values=fila)
        for i, item in enumerate(fila):
            col_width=tk.font.Font().measure(item)
            if self._tree.column(self._headers[i],width=None) <col_width:
                self._tree.column(self._headers[i],width=col_width)


#DEFINICION DEL APLICATIVO
class App(customtkinter.CTk):
    def __init__(self,ancho,alto,sistema):
        super().__init__()
        self.anch_pant=ancho
        self.alt_pant=alto
        self.sistema=sistema

        #Definicion del alto y ancho de la ventana
        self.ancho_vent=round(0.73*self.anch_pant)
        self.alto_vent=round(0.72*self.alt_pant)

        # configure window
        self.ruta_prog = os.getcwd()
        self.title("Software Deteccion")
        self.geometry(f'{self.ancho_vent}x{self.alto_vent}')
        self.resizable(width=False,height=False)

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # configuracion de la barra lateral
        self.sidebar_frame = customtkinter.CTkFrame(self, width=round(0.1*self.ancho_vent), corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)

        #Definicion de los botones de la barra lateral
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="MENU",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame,text='Inicio',command=self.frame_inicio)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame,text='Datos',command=self.frame_info)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame,text='Deteccion',command=self.frame_detect)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text='Salir', command=self.salir)
        self.sidebar_button_4.grid(row=4, column=0, padx=20, pady=10)

        #Definicion del cambio de apariencia
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appariencia:", anchor="w")
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Claro", "Oscuro"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=7, column=0, padx=20, pady=(10, 10))

        #Definicion del icono de la aplicacion
        #icono=tk.PhotoImage(file='./gorrion.jpg')
        #self.iconphoto(True,icono)

        #Valores por defecto
        customtkinter.set_appearance_mode("Light")
        self.frame_inicio()
        self.modo='Webcam'
        self.ruta=''
        self.caras = False
        self.guardar = False

    ##################################################################################################################################################################
    # ------------------------------------------- DEFINICION  DEL FRAME DE INICIO --------------------------------------#
    ##################################################################################################################################################################

    def frame_inicio(self):

        #Creacion del frame
        frame_ini = ctk.CTkFrame(self,width=round(0.464*self.ancho_vent),height=round(0.71*self.alto_vent),corner_radius=10)
        frame_ini.grid(row=0,column=1,pady=12,sticky='nsew')
        frame_ini.grid_columnconfigure((2, 3), weight=0)
        frame_ini.grid_rowconfigure((0, 1, 2), weight=1)

        #Titulo del frame
        titulo_ini=ctk.CTkLabel(frame_ini,text='BIENVENIDO A EL MULTIDETECTOR',font=ctk.CTkFont(size=28,weight='bold'))
        titulo_ini.place(x=80,y=30)

        textbox = customtkinter.CTkTextbox(frame_ini, width=600,height=400)
        textbox.place(x=30,y=80)
        textbox.insert('0.0',text='')

        frame_ini.tkraise()

##################################################################################################################################################################
# ------------------------------------------- DEFINICION  DEL FRAME DE DATOS --------------------------------------#
##################################################################################################################################################################
    def frame_info(self):

        #Creacion del frame
        self.frame_infor=ctk.CTkFrame(self,width=600,height=770,corner_radius=0)
        self.frame_infor.grid(row=0,column=1,pady=0,sticky='nsew')
        self.titulo_infor=ctk.CTkLabel(self.frame_infor,text='Data',font=ctk.CTkFont(size=24,weight='bold'))
        self.titulo_infor.place(x=450,y=10)
        
        #Tabla con los datos
        self.crear_tabla()

        #Creacion del mapa
        self.frame_infor.tkraise()

##################################################################################################################################################################
# ------------------------------------------- DEFINICION  DEL FRAME DEL DETECTOR --------------------------------------#
##################################################################################################################################################################


    def frame_detect(self):

        #Valores por defecto
        self.video = None
        self.ret = False
        modelo = YOLO('yolov8n.pt')
        anota_cubos = sv.BoxAnnotator(
            thickness=1,
            text_thickness=1,
            text_scale=0.4
        )
        self.lista_labels=['persona', 'bicicleta', 'coche', 'motocicleta', 'avion', 'autobus', 'tren', 'camion', 'barco', 'semaforo',
         'hidrante', 'senal de stop', 'parquimetro', 'banco', 'pajaro', 'gato', 'perro', 'caballo', 'oveja', 'vaca',
         'elefante', 'oso', 'cebra', 'jirafa', 'mochila', 'paraguas', 'bolso', 'corbata', 'maleta', 'frisbee', 'esquis',
         'tabla de snowboard', 'pelota de deportes', 'cometa', 'bate de beisbol', 'guante de beisbol', 'patineta',
         'tabla de surf', 'raqueta de tenis', 'botella', 'copa de vino', 'taza', 'tenedor', 'cuchillo', 'cuchara',
         'cuenco', 'platano', 'manzana', 'sandwich', 'naranja', 'brocoli', 'zanahoria', 'perro caliente', 'pizza',
         'dona', 'pastel', 'silla', 'sofa', 'planta en maceta', 'cama', 'mesa de comedor', 'inodoro', 'television',
         'portatil', 'raton', 'control remoto', 'teclado', 'telefono celular', 'microondas', 'horno', 'tostadora',
         'fregadero', 'refrigerador', 'libro', 'reloj', 'jarron', 'tijeras', 'osito de peluche', 'secador de pelo',
         'cepillo de dientes']


#------------------------------------------- DEFINICION DE LAS FUNCIONES DEL FRAME DEL DETECTOR --------------------------------------#
        
        #-----------
        #Funcion para la deteccion en video y webcam
        #-----------
        def deteccion(clase):
            inicio =time.time()
            if self.video == None:
                ...
            else:
                self.ret, frame = self.video.read()
                if self.ret == True and len(clase)>0 and self.checkbox_1.get()==1:

                    #Realizamos la deteccion
                    resultaados = modelo(frame, conf=0.3, classes=clase, verbose=False)[0]
                    detecciones = sv.Detections.from_yolov8(resultaados)
                    try:
                        array = [
                            array
                            for array, _,_, class_od, _ in detecciones if class_od == 0
                        ][0]
                        if self.caras == True and array is not None:
                            persona = frame[round(array[1]):round(array[3]), round(array[0]):round(array[2])]
                            cv2.imwrite(f'{self.ruta_prog}/Imagenes/{self.imagenes_dir}.jpg', persona)
                            self.imagenes_dir += 1
                    except IndexError:
                        print('Error en el indice durante el proceso')

                    nombres = [
                        f'{self.lista_labels[class_id]}'  # {confidence:0.2f}'
                        for _,_, confidence, class_id, _
                        in detecciones
                    ]

                    #Mostramos la informacion en el entry
                    num_per=len(nombres)
                    if self.max<num_per:
                        self.max=num_per
                    self.pers_vis_entr.delete(0,2)
                    self.pers_vis_entr.insert(0,str(num_per))
                    self.pers_vis_entr_max.delete(0,2)
                    self.pers_vis_entr_max.insert(0,str(self.max))

                    #Mostramos la imagen en el frame de deteccion
                    if self.switch_mostrar_caja.get()==1:
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

        #----------------
        #Funcion para realizar deteccion en funcion del modo de deteccion
        #------------------
        def main():
            etiq_video.place(x=round(0.017*self.ancho_detec), y=round(0.064*self.alto_detec))

            if (self.modo=='Video' or self.modo=='Webcam') and self.checkbox_1.get()==1:
                #clase = clases_para_detect()
                deteccion([0])
            elif self.modo=='Imagen'and self.checkbox_1.get()==1:
                clase = clases_para_detect()
                frame=cv2.imread(self.ruta)

                #Realizamos la deteccion sobre la imagen
                result=modelo(frame,conf=0.5, classes=clase,verbose=False)[0]
                detecciones = sv.Detections.from_yolov8(result)
                nombres = [
                    f'{self.lista_labels[class_id]}'  # {confidence:0.2f}'
                    for _,_, confidence, class_id, _
                    in detecciones
                ]

                #Mostramos la info en el entry
                num_per=len(nombres)
                self.pers_vis_entr.delete(0)
                self.pers_vis_entr.insert(0,str(num_per))
                self.pers_vis_entr_max.delete(0)
                self.pers_vis_entr_max.insert(0,str(num_per))

                #Preprocesamos la imagen y la mostramos en el frame de deteccion
                if self.switch_mostrar_caja.get()==1:
                        frame = anota_cubos.annotate(scene=frame, detections=detecciones, labels=nombres)
                frame = imutils.resize(frame, width=640, height=480)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img=img.resize((640,380),Image.ANTIALIAS)
                img = ImageTk.PhotoImage(image=img)

                #Aplicamos la imagen sobre la etiqueta
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

            #Comprobacion de los switches seleccionados
            lista=self.comprobar_switches()
            if len(lista)>=80:
                self.list=lista
            else:
                for eleme in lista:
                    self.list.append(dic[eleme])
            return self.list
        
        #-------------
        #Funcion que define la fuente de informacion para video y webcam
        #-------------
        def iniciar_web():

            #Definimos el máximo inicial
            self.max=0
            self.pers_vis_entr_max.delete(0)

            #Definimos la fuente de informacion en funcion del modo de deteccion
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

        #--------
        #Funcion para comprobar si el video existe y su formato es correcto
        #--------
        def comprobacion_video(ruta):
            valor = ruta.rsplit(sep='.',maxsplit=1)
            if valor[1].lower()==('mp4'or'mov'):
                return True
            else:
                print('\n El formato de video introducido no es válido\n'
                      'Los formatos permitidos son .mp4 .mov\n ')
                return False
                
        #------------
        #Funcion para comprobar si la imagen existe y su formato es correcto
        #------------
        def comprobacion_img(ruta):
            valor = ruta.rsplit(sep='.', maxsplit=1)
            if valor[1] == ('jpg' or 'png'or 'jpeg'):
                return True
            else:
                print('\n El formato de la imagen introducido no es válido\n '
                      'Los formatos permitidos son .jpg .png .jpeg\n')
                return False
                
        #----------
        #Funcion para la venta de confirmacion d einfo enviada
        #-----------------
        def info_env():

            #Definicion de la ventana
            vent=ctk.CTk()
            vent.geometry(f'{330}x{100}')
            vent.resizable(width=False,height=False)
            titulo=ctk.CTkLabel(vent,text='INFORMACION ENVIADA CORRECTAMENTE',font=ctk.CTkFont(size=14,weight='bold'))
            titulo.place(x=10,y=30)

            #Funcion para cerrar la ventana una vez pulsado ok
            def cerrar_vent():
                vent.destroy()

            #Definicion del boton de ok
            boton_ok=ctk.CTkButton(vent,text='Ok',command=cerrar_vent)
            boton_ok.place(x=100,y=65)

        #------------
        #Funcion para el envio de comandos en el fram de deteccion
        #-----------------
        def comandos():
            text_lst=self.entrada_comandos.get().split(sep=' ')

            #Comprobamos si hay texto escrito en el text entry
            if len(text_lst)==0:
                print('No se ha introducido ningun comando')
            elif len(text_lst)==2:
                texto = text_lst[0]
                ruta = text_lst[1]
                if os.path.exists(ruta)==False:
                    print('La ruta introducida no existe, introduce la ruta absoluta')
                else:

                    #Actualizamos los parametros en caso de que se trate de una imagen
                    if texto=='i':
                        self.modo='Imagen'
                        self.ruta = ruta
                        valor=comprobacion_img(self.ruta)
                        if valor==True:
                            textbox.delete('1.0','end')
                            textbox.insert('0.0', text=(f'Modo de deteccion: {self.modo}\n'
                                            f''))
                            info_env() 

                    #Actualizamos los parametros en caso de que se trate de un video
                    elif texto=='v':
                        self.modo = 'Video'
                        self.ruta=ruta
                        valor=comprobacion_video(self.ruta)
                        if valor==True:
                            textbox.delete('1.0', 'end')
                            textbox.insert('0.0', text=(f'Modo de deteccion: {self.modo}\n'
                                            f''))
                            info_env()
                        
                    else:
                        print('\nEl comando introducido no existe revisa el manual de comandos\n')

            elif len(text_lst)==1:
                texto = text_lst[0]
                if texto=='w':
                    self.modo = 'Webcam'
                    textbox.delete('1.0', 'end')
                    textbox.insert('0.0', text=(f'Modo de deteccion: {self.modo}\n'
                                            f''))
                    info_env()
                else:
                    print('\n El comando introducido no existe\n')
            elif len(text_lst)==5:
                self.database(text_lst)

        



        
        #------------------------
        #Funcion para la ventana de deteccion sobre pantalla
        #-------------------------
        def pantalla_detect():
            #Aseguramos que no se esta realizando deteccion en la ventana principal
            detener()

            #Minimizamos la ventana principal
            app.iconify()
            self.posicion=[]

            #Creacion de la ventana secundaria
            ventana_sec=ctk.CTk()
            ventana_sec.geometry(f'{300}x{300}')
            ventana_sec.resizable(width=False,height=False)
            ventana_sec.title('Deteccion sobre pantalla')

            #Frame general 
            frame_princ=ctk.CTkFrame(ventana_sec,width=250,height=235)
            frame_princ.place(x=25,y=0)

            #Personas vistas y cuadro de muestra
            pers_vis=ctk.CTkLabel(frame_princ,text='Personas vistas:')
            pers_vis.place(x=5,y=10)
            pers_vis_entr_2=ctk.CTkEntry(frame_princ,width=110)
            pers_vis_entr_2.place(x=120,y=10)

            #Modelo de deteccion para deteccion por pantalla
            modelo1=YOLO('yolov8m.pt')
            

            #--Funciones del frame
            def parar():
                #Para parar en caso de que se este ejecutanto por pantalla completa
                if pant_compl.get()==1:
                    pant_compl.deselect()
                
                #Para parar en caso de que se este ejecutando por region
                self.validar=False

            def ini_detect():
                
                if pant_compl.get()==1 :

                    #Toma de la captura y preprocesamiento de la imagen
                    img=pyautogui.screenshot()
                    img=np.array(img)
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

                    #Ejecucion de la prediccion
                    resultaados = modelo1(img, conf=0.5, classes=[0], verbose=False)[0]
                    detecciones = sv.Detections.from_yolov8(resultaados)
                    num_pers=len(detecciones)
                    pers_vis_entr_2.delete(0,2)
                    pers_vis_entr_2.insert(0,str(num_pers))
                    print('ejecutando')

                    pers_vis_entr_2.after(2,ini_detect)
                elif len(self.posicion)==4 and self.validar==True:
                    img=pyautogui.screenshot(region=(self.posicion[0],self.posicion[1],self.posicion[2]-self.posicion[0],self.posicion[3]-self.posicion[1]))
                    img=np.array(img)
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


                    #Ejecucion de la prediccion
                    resultaados = modelo1(img, conf=0.5, classes=[0], verbose=False)[0]
                    detecciones = sv.Detections.from_yolov8(resultaados)
                    num_pers=len(detecciones)
                    pers_vis_entr_2.delete(0,2)
                    pers_vis_entr_2.insert(0,str(num_pers))
                    print('ejecutando')
                    pers_vis_entr_2.after(2,ini_detect)
                else:
                    print('Seleccione una region o marque pantalla completa')
                
            #Funcion para la eleccion del area de deteccion
            def eleccion1():
                if pant_compl.get()==1:
                    pant_compl.deselect()
                self.validar=False

                #Crea una ventana transparente sobre la que haciendo drag con el raton se elige la zona
                vent_transp=tk.Tk()
                vent_transp.geometry(f'{self.winfo_screenwidth()}x{self.winfo_screenheight()}')
                vent_transp.wait_visibility(self)
                vent_transp.wm_attributes('-alpha',0.3)
                
                vent_transp.title('Click Me')

                #Obtenemos las coordenadas al inicio del drag
                def ini_drag(event):
                    self.x_1=event.x
                    self.y_1=event.y

                #Obtenemos las coordenas del final del drag
                def fin_drag(event):
                    self.x_2=event.x
                    self.y_2=event.y

                    #En caso de que el punto se haya seleccionado correctament
                    if self.x_1==self.x_2 and self.y_1==self.y_2 or self.y_2<self.y_1 or self.x_2<self.x_1:
                        ...
                    else:
                        self.posicion=[self.x_1,self.y_1,self.x_2,self.y_2]

                        #Introduccion de los valores de las coordenadas en la entradas correspondientes
                        x1_entry.delete(0,4)
                        x1_entry.insert(0,str(self.posicion[0]))
                        x2_entry.delete(0,4)
                        x2_entry.insert(0,str(self.posicion[2]))
                        y1_entry.delete(0,4)
                        y1_entry.insert(0,str(self.posicion[1]))
                        y2_entry.delete(0,4)
                        y2_entry.insert(0,str(self.posicion[3]))
                        
                        #Para permitir la deteccion ya que el punto elegido es correcto
                        self.validar=True

                        #Dibujo del recuadro elegido para la deteccion
                        #canvas=tk.Canvas(vent_transp,width=self.winfo_screenwidth(),height=self.winfo_screenheight(),bg='black')
                        #vent_transp.wait_visibility(canvas)
                        #vent_transp.wm_attributes('-alpha',0.6)
                        #canvas.create_rectangle(self.x_1,self.y_1,self.x_2,self.y_2,outline='white')
                        #canvas.pack()
                        vent_transp.destroy()

                #Indica que al realizar acciones con el click izquierdo del mouse conduza a su respectiva funcion
                vent_transp.bind('<ButtonPress-1>', ini_drag)  
                vent_transp.bind('<ButtonRelease-1>',fin_drag)

                vent_transp.mainloop()
            
            #Funcion para el boton de salir
            def exit():
                ventana_sec.destroy()
                
            #Botones principales
            bot_inciar=ctk.CTkButton(ventana_sec,text='Iniciar',command=ini_detect,width=75)
            bot_inciar.place(x=15,y=250)
            bot_inciar=ctk.CTkButton(ventana_sec,text='Detener',command=parar,width=75)
            bot_inciar.place(x=100,y=250)
            bot_parar=ctk.CTkButton(ventana_sec,text='Salir',command=exit,width=75)
            bot_parar.place(x=190,y=250)
            bot_elec_area=ctk.CTkButton(ventana_sec,text='Elegir recuadro',command=eleccion1)
            bot_elec_area.place(x=90,y=200)
            pant_compl=ctk.CTkCheckBox(ventana_sec,text='Pantalla completa')
            pant_compl.place(x=90,y=170)

            #Definicion textos y entradas de los puntos
            x1=ctk.CTkLabel(frame_princ,text='X1')
            x1.place(x=5,y=40)
            x1_entry=ctk.CTkEntry(frame_princ,width=50)
            x1_entry.place(x=25,y=40)
            y1=ctk.CTkLabel(frame_princ,text='Y1')
            y1.place(x=110,y=40)
            y1_entry=ctk.CTkEntry(frame_princ,width=50)
            y1_entry.place(x=130,y=40)
            x2=ctk.CTkLabel(frame_princ,text='X2')
            x2.place(x=5,y=70)
            x2_entry=ctk.CTkEntry(frame_princ,width=50)
            x2_entry.place(x=25,y=70)
            y2=ctk.CTkLabel(frame_princ,text='Y2')
            y2.place(x=110,y=70)
            y2_entry=ctk.CTkEntry(frame_princ,width=50)
            y2_entry.place(x=130,y=70)


        #-----
        #Funcion para abrir el navegador de archivos
        #------
        def explorador():
            filename=()

            filetypes=(('Archivos de video',('*.mp4','*.mov')),
                        ('Archivos de imagen',('*.jpg','*jpeg','*.png')))
            
            filename = tk.filedialog.askopenfilename(initialdir = "/home/zelenza",
                                          title = "Elige un archivo",
                                          filetypes=filetypes)
            if filename !=() and filename!='':
                terminacion=filename.split(sep='.')[1]
                if terminacion=='mp4' or terminacion=='mov':
                    self.modo='Video'
                    self.ruta=filename
                    textbox.delete('1.0', 'end')
                    textbox.insert('0.0', text=(f'Modo de deteccion: {self.modo}\n'
                                                f''))
                    info_env()
                elif terminacion=='jpeg' or terminacion=='jpg' or terminacion=='png':
                    self.modo='Imagen'
                    self.ruta=filename
                    textbox.delete('1.0', 'end')
                    textbox.insert('0.0', text=(f'Modo de deteccion: {self.modo}\n'
                                                f''))
                else:
                    print('El archivo elegido no tiene el formato adecuado')



        ##########################################################################################
                    #Definicion del resto de widgets del frame detector
        ##########################################################################################

        #Definicion del ancho y alto del frame de deteccion
        self.ancho_detec=round(0.43*self.ancho_vent)
        self.alto_detec= self.alto_vent

        #Creacion del frame
        frame_dete = ctk.CTkFrame(self,width=self.ancho_detec,height=self.alto_detec,corner_radius=0)
        frame_dete.grid(row=0,column=1,pady=12,sticky='nsew')
        self.titulo_dete=ctk.CTkLabel(frame_dete,text='Deteccion',font=ctk.CTkFont(size=24,weight='bold'))
        self.titulo_dete.place(x=round(0.75*self.ancho_detec),y=round(0.013*self.alto_vent))

        #Definicion etiqueta imagenes
        etiq_video = tk.Label(frame_dete)
        etiq_video.place(x=round(0.017*self.ancho_detec), y=round(0.064*self.alto_detec))
        self.boton_inciar_det=ctk.CTkButton(frame_dete,text='Iniciar',command=iniciar_web)
        self.boton_inciar_det.place(x=round(0.3*self.ancho_detec),y=round(0.55*self.alto_detec))
        self.boton_detener_det=ctk.CTkButton(frame_dete,text='Detener',command=detener)
        self.boton_detener_det.place(x=round(0.58*self.ancho_detec),y=round(0.55*self.alto_detec))

        #Frame para los switches
        scrollable_frame = customtkinter.CTkScrollableFrame(frame_dete, label_text="Opciones",label_font=ctk.CTkFont(size=16,weight='bold'))
        scrollable_frame.place(x=round(1.63*self.ancho_detec),y=round(0.013*self.alto_detec))
        scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame_switches = []
        self.lista_objetos=self.lista_labels
        self.lista_objetos.insert(0,'Todos')
        i=0
        for elem in self.lista_objetos:
            self.switch = ctk.CTkSwitch(master=scrollable_frame, text=f"{elem}",command=soltar_listo)
            self.switch.grid(row=i, column=0, padx=10, pady=(0, 20))
            i+=1
            self.scrollable_frame_switches.append(self.switch)
        self.checkbox_1 = customtkinter.CTkCheckBox(frame_dete,text='Seleccion preparada',command=self.comprobar_switches)
        self.checkbox_1.place(x=round(1.63*self.ancho_detec),y=round(0.35*self.alto_detec))

        #Barra introduccion de comandos
        self.entrada_comandos=ctk.CTkEntry(frame_dete,placeholder_text='Introduce un comando',width=round(0.92*self.ancho_detec))
        self.entrada_comandos.place(x=round(0.017*self.ancho_detec),y=round(0.9*self.alto_detec))
        img=Image.open('./carpeta.png')
        img=ctk.CTkImage(light_image=img,dark_image=img)
        
        self.expl_arch=ctk.CTkButton(frame_dete,image=img,width=30,height=30,text='',command=explorador)
        self.expl_arch.place(x=round(0.94*self.ancho_detec),y=round(0.9*self.alto_detec))
        self.boton_comandos=ctk.CTkButton(frame_dete,text='Enviar',command=comandos)
        self.boton_comandos.place(x=self.ancho_detec,y=round(0.9*self.alto_detec))

        #Frame de informacion
        self.info_frame_ancho=round(0.45*self.ancho_detec)
        self.info_frame_alto=round(0.61*self.alto_detec)
        self.info_frame=ctk.CTkFrame(frame_dete,width=self.info_frame_ancho,height=self.info_frame_alto,corner_radius=5)
        self.info_frame.place(x=round(1.13*self.ancho_detec),y=round(0.064*self.alto_detec))
        self.titulo_info=ctk.CTkLabel(self.info_frame,text='Informacion', font=ctk.CTkFont(size=16,weight='bold'))
        self.titulo_info.place(x=round(0.16*self.info_frame_ancho),y=round(0.02*self.info_frame_alto))
        self.pers_vis=ctk.CTkLabel(self.info_frame,text='Personas vistas:')
        self.pers_vis.place(x=round(0.04*self.info_frame_ancho),y=round(0.104*self.info_frame_alto))
        self.pers_vis_entr=ctk.CTkEntry(self.info_frame)
        self.pers_vis_entr.place(x=round(0.47*self.info_frame_ancho),y=round(0.104*self.info_frame_alto))
        self.pers_vis_max=ctk.CTkLabel(self.info_frame,text='Personas máximo vistas:')
        self.pers_vis_max.place(x=round(0.04*self.info_frame_ancho),y=round(0.17*self.info_frame_alto))
        self.pers_vis_entr_max=ctk.CTkEntry(self.info_frame, width=round(0.37*self.info_frame_ancho))
        self.pers_vis_entr_max.place(x=round(0.62*self.info_frame_ancho),y=round(0.17*self.info_frame_alto))

        #Frame modos y rutas y botones
        textbox = customtkinter.CTkTextbox(frame_dete, width=round(0.41*self.ancho_detec),height=round(0.19*self.alto_detec))
        textbox.place(x=round(1.63*self.ancho_detec), y=round(0.4*self.alto_detec))
        textbox.insert('0.0', text=(f'Modo de deteccion: {self.modo}\n'
                                    f''))
        self.switch_guardar = ctk.CTkSwitch(frame_dete, text='Guardar output')
        self.switch_guardar.place(x=round(1.63*self.ancho_detec), y= round(0.6*self.alto_detec) )
        self.switch_caras = ctk.CTkSwitch(frame_dete, text='Extraer personas')
        self.switch_caras.place(x=round(1.63*self.ancho_detec), y=round(0.64*self.alto_detec))
        self.switch_mostrar_caja = ctk.CTkSwitch(frame_dete, text='Dibujar recuadro')
        self.switch_mostrar_caja.place(x=round(1.63*self.ancho_detec), y=round(0.68*self.alto_detec))

        #Boton deteccion sobre pantalla
        self.detect_pant=ctk.CTkButton(frame_dete,text='Deteccion sobre pantalla'
                                       ,command=pantalla_detect)
        self.detect_pant.place(x=round(1.63*self.ancho_detec),y=round(0.74*self.alto_detec))

        frame_dete.tkraise()

##################################################################################################################################################################
# ------------------------------------------- DEFINICION  DE OTRAS FUNCIONES GENERALES --------------------------------------#
##################################################################################################################################################################

    #--------
    #Encargada de el cambio de apariencia de la ventana
    #---------
    def change_appearance_mode_event(self, new_appearance_mode: str):
        dic={'Claro':'Light','Oscuro':'Dark'}
        valor = dic[new_appearance_mode]
        ctk.set_appearance_mode(valor)

    #----------
    #Funcion encargada de comprobar los switches seleccionados
    #----------
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
    
    #-------
    #Funcion para destruir la ventana y eliminar toda ejecucion de la aplicacion
    #------
    def salir(self):
        lista=[]
        abre=os.popen('ps')
        lee=abre.readlines()
        del lee[0]
        for i in lee:
            lista.append(i.strip())
        for elem in lista:
            texto=elem.split(sep=' ')
            if (texto[6]=='python3'):
                pid= texto[0]
        os.kill(int(pid),signal.SIGKILL)
        app.destroy()
        os.kill(int(pid))

    #--------
    #Funcion encargada de crear la tabla para el frame de datos
    #---------
    def crear_tabla(self):

        #Creacion del frame datos-graficas
        tabla_info=Table(self.frame_infor,title='',headers=(u'Fecha',u'Dispositivo',u'Nº Personas',u'Tiempo_ejec',u'Longitud',u'Latitud'))
        tabla_info.place(x=0,y=500)
        conexion= psycopg2.connect(host='localhost',database="detecciones", user="zelenza", password="zelenza2023")
        self.cursor1=conexion.cursor()
        self.cursor1.execute("select fecha,dispositivo,numero_personas,tiempo_ejecucion,longitud,latitud from detecciones")
        fechas={}
        for elem in self.cursor1:
            tabla_info.anadir_fila(elem)
            if (str(elem[0]).split(sep=' ',maxsplit=2)[0]) not in fechas:
                fechas[str(elem[0]).split(sep=' ',maxsplit=2)[0]]=1
            else:
                fechas[str(elem[0]).split(sep=' ',maxsplit=2)[0]]= fechas[str(elem[0]).split(sep=' ',maxsplit=2)[0]]+1
        
        etiquetas=list(fechas.keys())
        x=np.arange(1,len(etiquetas)+1)

        y=list(fechas.values())
        self._fig1,self._ax1=plt.subplots()
        self._ax1.set_title('Cantidad de detecciones')

        self._ax1.bar(x=x,height=y,tick_label=etiquetas)

        self._fig1_canvas=matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(
            self._fig1,master=self.frame_infor
        )


    
    def database(self,lista):
        print('Cargando datos...')
        if lista[0]=='pg':
            try:
                conexion= psycopg2.connect(host=lista[1],database=lista[2], user=lista[3], password=lista[4])
                self.cursor1=conexion.cursor()
                self.cursor1.execute("select * from imagenes;")
                for elem in self.cursor1:
                    print(elem)
                print('Datos recibidos!')
            except psycopg2.OperationalError:
                print('Se ha producido un error durante la conexion con la base de datos')

##################################################################################################################################################################
# ------------------------------------------- FIN DEL PROGRAMA --------------------------------------#
##################################################################################################################################################################

#El siguiente codigo lanza la aplicacion al ejecutar el script
if __name__ == "__main__":
    app = App(pyautogui.size().width,pyautogui.size().height,platform.system())
    app.mainloop()
