# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:33:26 2021

Este codigo hará la transformación de archivos pcap 

01.- Realiza una lista de archivos pcap dentro de todo el directorio, incluyendo subcarpetas
02.- Crear las imagenes a través de 3fex usando consola de comandos
03.- Crear carpetas junto a los archivos pcap con el nombre de cada archivo pcap
04.- Se mueven las imagenes creadas a cada carpeta respectiva a cada archivo pcap.
    Se le cambia el nombre al siguiente: "nombre_archivo_pcap - "numero_de_imagen
"""

import platform,os
import pandas as pd
import cv2

class ListEmpty(Exception):
    """Exception por una lista vacía

    """

    def __init__(self,tipo_archivo, mensaje="El directorio no contiene archivos de tipo "):
        self.tipo_archivo = tipo_archivo
        self.mensaje = mensaje
        super().__init__(self.mensaje)

    def __str__(self):
        return f'{self.mensaje}{self.tipo_archivo} en el directorio -> {os.getcwd()}'
    


if platform.system() == "Linux": #Comprobar el tipo de sistema que es.
    """--------------------------------- Paso 1 ---------------------------------"""
    #01.- Realiza una lista de archivos pcap dentro de todo el directorio, incluyendo subcarpetas
    
    path = os.getcwd()
    list_of_files = list()
    pcap_fail = list()
    list_of_u_files = list()
    [[list_of_files.append(os.path.join(root,file)) for file in files]
     for root, folders, files in os.walk(path)]
            
    pcap_list = [i for i in list_of_files if i[-5:] == ".pcap"]
    
    if pcap_list == []: raise ListEmpty("pcap")
    
    #lista de todos archivos pcap dentro del directorio
    for pcap_file in pcap_list:
        csv_file = pcap_file[:-5] + ".csv"
        if os.path.isfile(csv_file):
            continue
        """--------------------------------- Paso 2 ---------------------------------"""
        #02.- Crear las imagenes a través de 3fex usando consola de comandos
        try:
            """
            comando a usar sudo snort -c /etc/snort/snort.conf -r traza_original.pcap crea archibo tipo U para agregar a 3fex y agregar -p 1 para prioridades de tipo 1
            sudo 3fex -r traza_original.pcap -f traza_original.csv -o 8 -noips -u /var/log/snort/snort.log.1625018634 -p 1
            """
            u_command = "sudo snort -c /etc/snort/snort.conf -r '" + pcap_file +"'"
            print("\nu_command: ", u_command)
            os.system(u_command) #Creando archivo U
            list_of_u_files = os.listdir("/var/log/snort/")
            antiguo_u_file = sorted(list_of_u_files)[-1] #Archivo más recientemente creado
            u_file = antiguo_u_file + ".x"
            name = pcap_file.split("/")
            name = name[-1][:-5]
            directorio = pcap_file[:-5] #os.path.join(pcap_file[:-5],name)
            print("\ndirectorio: ", directorio)
            mov_u_file = "sudo mv " + "/var/log/snort/" + antiguo_u_file + " " + directorio + "_" + u_file
            print("\nMoviendo archivo u al directorio: ", mov_u_file)
            os.system(mov_u_file)
            
            command = "sudo 3fex -r '" + pcap_file + "' -f '" + csv_file + "' -o 1 -noips -u '" + directorio + "_" + u_file + "' -p 1"
            #Ejemplo   sudo 3fex -r traza_original.pcap -f traza_original.csv -o 8 -noips -u /var/log/snort/snort.log.1625018634 -p 1
            print("\nIniciando creación de imagenes para el archivo: ",pcap_file)
            print("\ncommand: ",command)
            os.system(command)
        except:
            print("Comando inválido: ", command)
            print("Elimine las imagenes creadas e intente nuevamente")
            continue
        
        imagenes = [i[:-4] for i in os.listdir(os.getcwd()) if i [-4:] == ".jpg"]
        imagenes = [str(i) + ".jpg" for i in range(len(imagenes))]
        
        if imagenes == []:
            pcap_fail.append(pcap_file)
            print("No se pudo ejecutar el archivo: ", pcap_file)
            continue
        """--------------------------------- Paso 3 ---------------------------------"""
        #03.- Crear carpetas junto a los archivos pcap con el nombre de cada archivo pcap
        try:
            folder = "mkdir -p " + pcap_file[:-5]
            print("Creando la carpeta: ", folder)
            os.system(folder)
        except:
            print("Comando inválido: ", folder)
            continue
        
        """--------------------------------- Paso 4 ---------------------------------"""
        #04.- Se mueven las imagenes creadas a cada carpeta respectiva a cada archivo pcap.
        contador = 0
        print("csv: " ,csv_file)
        df = pd.read_csv(csv_file, index_col=0)
        values = df["label"]
        for i in imagenes:
            try:
                # Crear dataset
                mover_archivo_comando = "mv " + i + " " + os.path.join(pcap_file[:-5],name) + "-" + i[:-4] + "_" + str(values[contador]) + ".jpg"
                contador += 1
                print("\nMoviendo imagenes generadas al directorio: ", directorio)
                print("archivo: ", mover_archivo_comando)
                os.system(mover_archivo_comando)
            except:
                print("Comando inválido: ", mover_archivo_comando)
                continue
else:
    print("Este sistema operativo no es Linux")