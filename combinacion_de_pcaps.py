import os
import pandas as pd

pcap_1 = "esc-02-Mixed-traffic.pcap"
pcap_2 = "botnet-capture-20110811-neris-refiltered-malicious.pcap"
csv_name = "botnet.csv"

def comb_pcaps(pcap_1, pcap_2, csv_name, folder_name = "imgs"):
    
    if pcap_1[-5:] != ".pcap" or pcap_2[-5:] != ".pcap" or csv_name[-4:] != ".csv":
        raise Exception("incorrect extensions. Please check files name")
    
    # Combine .pcaps
    comando = "3fex -r " + pcap_1 + " -b " + pcap_2 + " -f " + csv_name + " -p 1 -o 1 -noips"
    os.system(comando)
    
    # Creating folder for new images
    try:
        os.mkdir(folder_name)
    except:
        pass
    
    df = pd.read_csv(csv_name, index_col = 1)
    values = df["label"][1:]
    
    archivos_ordenados = []
    index = 0
    
    for root, folders, files in os.walk(os.getcwd()):
        for file in files:
            if file[-4:] != ".jpg":
                continue
            file_name = str(index) + ".jpg"
            archivos_ordenados.append(file_name)
            index += 1
        
    for arch in archivos_ordenados:
        name_file = os.path.basename(arch)[:-4] + "_" + str(int(values[int(os.path.basename(arch)[:-4])])) + ".jpg"
        mov_file_com = "sudo mv " + os.path.join(root,arch) + " " + os.path.join(root, folder_name, name_file)
        os.system(mov_file_com)
        
""" backup
    archivos = []
    archivos_ordenados = []
    
    for root, folders, files in os.walk(os.getcwd()):
        for file in files:
            if file[-3:] == "jpg":
                archivos.append(file)
    
    for i in range(len(archivos)):
        file_name = str(i) + ".jpg"
        archivos_ordenados.append(file_name)
        
    for arch in archivos_ordenados:
        name_file = os.path.basename(arch)[:-4] + "_" + str(int(values[int(os.path.basename(arch)[:-4])])) + ".jpg"
        mov_file_com = "sudo mv " + os.path.join(root,arch) + " " + os.path.join(root, "imgs", name_file)
        os.system(mov_file_com)
"""