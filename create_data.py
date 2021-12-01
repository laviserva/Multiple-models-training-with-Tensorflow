# -*- coding: utf-8 -*-
"""
1. Se revisa en el path main si existe el archivo pickle y se carga en memoria
   El archivo pickle es donde está almacenado toda la dataset a usar.
2. Se crea el archivo pickle para evitar cargar a memoria a cada rato toda la dataset.
   hacerlo de manera unica y rápida
3. Se normaliza la data y randomizar
4. Prepara en training y testing.

"""


import os
import numpy as np

from sklearn.utils import shuffle

def create_images(in_paths = [os.getcwd()], in_pickle =os.getcwd(), out_pickle = os.getcwd(), out_name_pickle = "data_pcaps_files.pickle",
                  pickle_file = "data_pcaps_files.pickle", normalize = True, randomize = True, width = 1, height = 280):
    """
    Parameters
    ----------
    in_paths : 
        DESCRIPTION. El path de entrada para multiple paths, el defecto es el archivo py, os.getcwd().
                     
    in_pickle : 
        DESCRIPTION. Se introducie el path del pickle en caso de que se encuentre en algun otro path
                     El valor predeterminado es os.getcwd().
        
    out_pickle : 
        DESCRIPTION. Es la salida deseada del archivo pickle. por defecto se exporta en os.getcwd()
                     El cual es el directorio del archivo py.
     out_name_pickle : 
        DESCRIPTION. Es la nombre deseado del archivo pickle. por defecto se exporta en os.getcwd()
                     El cual es el directorio del archivo py.
        
    normalize : 
        DESCRIPTION. Si este valor está activo, la data se normaliza.
                     El valor por default es True.
        
    randomize : 
        DESCRIPTION. Si este valor está activo, la data se randomiza.
                     El valor por default es True.
        
    Returns
    -------
    data :
        Devuelve una tupla, el valor 0 son los datos de entrenamiento en formato numpy.
        El valor 1 de la tupla son las etiquetas.
    """
    
    import pickle
    
    if list is not type(in_paths):
        in_paths = [in_paths]
        
    in_pickle = [in_pickle]
    pickle_path = os.path.join(in_pickle[0], pickle_file)
    #in_pickle[0] + pickle_file


    if os.path.isfile(pickle_path):
        """ --------------------------------------- paso 1 ---------------------------------------"""
        #Cargar data
        print("Archivo pickle encontrado. ")
        print("- - - Desempaquetando - - -")
        print("Archivo: ", pickle_path)
        
        pickle_out = open(pickle_path,"rb")
        data = pickle.load(pickle_out)
        pickle_out.close()

        print("------ Desempaquetado -----")
        
        data_images = np.array(data[0])
        data_labels = np.array(data[1])
        
        contador = 0
        for i in data_labels:
            if i == 1:
                contador += 1
        print("De un total de {} imagenes, solo {} tienen etiqueta 1 y {} etiqueta 0".format(
            len(data_labels), contador, len(data_labels) - contador ))
        print("{0:.5f}% positivos".format(contador/len(data_labels)))
        print("{0:.5f}% negativos".format((len(data_labels) - contador)/ len(data_labels)))
        
        return [data_images,data_labels]
        
    else:
        import cv2
        """ --------------------------------------- paso 2 ---------------------------------------"""
        #Hacer para varios paths y colocar el pickle en donde mejor convenga
        #crear data
        
        data_images = list()
        data_labels = list()
        
        dim = (height, width)
        for i in in_paths:  
            for root, folders, files in os.walk(i):
                print(root)
                
                for file in files:
                    if file[-4:] == ".jpg" or file[-4:] == ".png":
                        path_img = os.path.join(root,file)
                        img = cv2.imread(path_img,0)
                        #img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                        if img.shape == ( width, height):
                            if img is not None and (file[-5] == "0" or file[-5] == "1"):
                                data_labels.append(int(file[-5]))
                                data_images.append(img)
        
        data_images = np.array(data_images)
        data_labels = np.array(data_labels)
        
        data_images = data_images[:,:,:,np.newaxis]
        """
        contador = 0
        for i in data_labels:
            if i == 1:
                contador += 1
        print("De un total de {} imagenes, solo {} tienen etiqueta 1 y {} etiqueta 0".format(
            len(data_labels), contador, len(data_labels) - contador ))
        print("{0:.5f}% positivos".format(contador/len(data_labels)))
        print("{0:.5f}% negativos".format((len(data_labels) - contador)/ len(data_labels)))
        """
        if normalize == True:
            data_images =  np.divide(data_images, 255.0)
        
        if randomize == True:
            data_images, data_labels = shuffle(data_images, data_labels)
            
            
        data = [data_images,data_labels]
        
        out_pickle = out_pickle + "\\" + out_name_pickle
    
        print("Guardando el archivo: ", out_pickle) 
    
        pickle_out = open(out_pickle,"wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()
        
        return data
#lista_pcaps_dirs = [r"D:\Dataset\trazas", r"C:\Users\Private Richi\Documents\ss\trazas\pcap_files"]

#data = create_images(in_paths = lista_pcaps_dirs, out_pickle = r"C:\Users\Private Richi\Documents\ss\trazas")