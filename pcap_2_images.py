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

    def __init__(self,tipo_archivo, mensaje="No pcap files in this directory"):
        self.tipo_archivo = tipo_archivo
        self.mensaje = mensaje
        super().__init__(self.mensaje)

    def __str__(self):
        return f'{self.mensaje}{self.tipo_archivo} -> {os.getcwd()}'
    


if platform.system() == "Linux":
    
    path = os.getcwd()
    list_of_files = list()
    pcap_fail = list()
    list_of_u_files = list()
    [[list_of_files.append(os.path.join(root,file)) for file in files]
     for root, folders, files in os.walk(path)]
            
    pcap_list = [i for i in list_of_files if i[-5:] == ".pcap"]
    
    if pcap_list == []: raise ListEmpty("pcap")
    
    for pcap_file in pcap_list:
        csv_file = pcap_file[:-5] + ".csv"
        if os.path.isfile(csv_file):
            continue
        try:
            """
            example command:
                sudo 3fex -r traza_original.pcap -f traza_original.csv -o 8 -noips -u /var/log/snort/snort.log.1625018634 -p 1
            """
            
            u_command = "sudo snort -c /etc/snort/snort.conf -r '" + pcap_file +"'"
            os.system(u_command)
            list_of_u_files = os.listdir("/var/log/snort/")
            antiguo_u_file = sorted(list_of_u_files)[-1] # most recent file created
            u_file = antiguo_u_file + ".x"
            name = pcap_file.split("/")
            name = name[-1][:-5]
            directorio = pcap_file[:-5] # os.path.join(pcap_file[:-5],name)
            mov_u_file = "sudo mv " + "/var/log/snort/" + antiguo_u_file + " " + directorio + "_" + u_file
            os.system(mov_u_file)
            
            command = "sudo 3fex -r '" + pcap_file + "' -f '" + csv_file + "' -o 1 -noips -u '" + directorio + "_" + u_file + "' -p 1"
            """
            Example command:
                sudo 3fex -r traza_original.pcap -f traza_original.csv -o 8 -noips -u /var/log/snort/snort.log.1625018634 -p 1
            """
            os.system(command)
        except:
            print("Invalid command: ", command)
            continue
        
        imagenes = [i[:-4] for i in os.listdir(os.getcwd()) if i [-4:] == ".jpg"]
        imagenes = [str(i) + ".jpg" for i in range(len(imagenes))]
        
        if imagenes == []:
            pcap_fail.append(pcap_file)
            print("Error loading pcap file: ", pcap_file)
            continue
        try:
            folder = "mkdir -p " + pcap_file[:-5]
            os.system(folder)
        except:
            print("Invalid command: ", folder)
            continue
        
        contador = 0
        print("csv: " ,csv_file)
        df = pd.read_csv(csv_file, index_col=0)
        values = df["label"]
        for i in imagenes:
            try:
                mover_archivo_comando = "mv " + i + " " + os.path.join(pcap_file[:-5],name) + "-" + i[:-4] + "_" + str(values[contador]) + ".jpg"
                contador += 1
                os.system(mover_archivo_comando)
            except:
                print("Invalid command: ", mover_archivo_comando)
                continue
else:
    print("This S.O. is not Linux")