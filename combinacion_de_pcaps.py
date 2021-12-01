import os
import pandas as pd

img_list = []

pcap_1 = "esc-02-Mixed-traffic.pcap"
pcap_2 = "botnet-capture-20110811-neris-refiltered-malicious.pcap"
csv_file = "botnet.csv"

comando = "3fex -r " + pcap_1 + " -b " + pcap_2 + " -f " + csv_file + " -p 1 -o 1 -noips"

os.system(comando)
try:
    os.mkdir("imgs")
except:
    pass

print("iniciar csv")
df = pd.read_csv(csv_file, index_col = 1)
print("csv leido")
values = df["label"][1:]

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
    mov_file_com = "sudo mv " + os.path.join(root,arch) + " " + os.path.join(os.path.join(root, "imgs"), name_file)
    os.system(mov_file_com)