import os
import numpy as np

from sklearn.utils import shuffle

def create_images(in_paths = [os.getcwd()], in_pickle =os.getcwd(), out_pickle = os.getcwd(), out_name_pickle = "data_pcaps_files.pickle",
                  pickle_file = "data_pcaps_files.pickle", normalize = True, randomize = True, width = 1, height = 280):
    """
    Parameters
    ----------
    in_paths : 
        DESCRIPTION. input path, os.getcwd().
                     
    in_pickle : 
        DESCRIPTION. Input Pickle path
        
    out_pickle : 
        DESCRIPTION. Output Pickle path
        
     out_name_pickle : 
        DESCRIPTION. Pickle Default name
        
    normalize : 
        DESCRIPTION. Normalize data if true
        
    randomize : 
        DESCRIPTION. Randomize if true
        
    Returns
    -------
    data :
        Return data
    """
    
    import pickle
    
    if list is not type(in_paths):
        in_paths = [in_paths]
        
    in_pickle = [in_pickle]
    pickle_path = os.path.join(in_pickle[0], pickle_file)
    #in_pickle[0] + pickle_file


    if os.path.isfile(pickle_path):
        pickle_out = open(pickle_path,"rb")
        data = pickle.load(pickle_out)
        pickle_out.close()
        
        data_images = np.array(data[0])
        data_labels = np.array(data[1])
        
        contador = 0
        for i in data_labels:
            if i == 1:
                contador += 1
        print("From {} images.\nLabel 1: {}\nLabel 0: {}".format(
            len(data_labels), contador, len(data_labels) - contador ))
        print("{0:.5f}% positives".format(contador/len(data_labels)))
        print("{0:.5f}% negatives".format((len(data_labels) - contador)/ len(data_labels)))
        
        return [data_images,data_labels]
        
    else:
        import cv2
        
        data_images = list()
        data_labels = list()
        
        #dim = (height, width)
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
        
        if normalize == True:
            data_images =  np.divide(data_images, 255.0)
        
        if randomize == True:
            data_images, data_labels = shuffle(data_images, data_labels)
            
            
        data = [data_images,data_labels]
        
        out_pickle = out_pickle + "\\" + out_name_pickle
    
        print("Saving file: ", out_pickle) 
    
        pickle_out = open(out_pickle,"wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()
        
        return data