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