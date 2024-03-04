# -*- coding: utf-8 -*-
import logging
import json
import sys
from downloader import  get_url
import requests
from PIL import Image
from io import BytesIO
import os
from downloader import  get_url
from hdfs import InsecureClient  # Aggiunta l'importazione per HDFS
# Inizializza il client HDFS
hdfs_client = InsecureClient("http://node-master:50070")
out_dir="/"


def download_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Converti il contenuto dell'immagine in un oggetto Image
            image = Image.open(BytesIO(response.content))
            
            return image
        
    except Exception as e:
        # Gestisci eventuali eccezioni nel download dell'immagine
        print(f"Errore nel download dell'immagine: {str(e)}")
        return None
    

def save_image(image, output_dir, filename):
    
     if image is not None:
        # Salva l'immagine nella directory
        try:
            with hdfs_client.write(f"{output_dir}/{filename}", overwrite=True) as writer:
             image.save(writer, format="JPEG")
        except Exception as e:
            print(f"Errore nel salvataggio: {str(e)}")
          
       
def get_urls(line):
    datain = json.loads(line)
    server = datain['server']
    start_col = datain['pos1y']+datain['start_col']
    end_col = start_col + datain['subsize']
    start_row = datain['pos1x']+datain['start_row']
    end_row = start_row + datain['subsize']
    urls = [get_url(datain['server'], i, j, datain['zoom'], datain['style']) \
        for j in range(start_col,end_col) \
            for i in range(start_row, end_row)]
    
    return urls,datain



# ---------------------------------------------------------
if __name__ == '__main__':
    
    for line in sys.stdin:

        urls,datain=get_urls(line)
        tiles_dir=datain['tiles_dir']
        datain['tiles_dir']=f"{out_dir}{tiles_dir}"
        hdfs_client.makedirs(datain['tiles_dir'], permission=755)

        start_col = datain['start_col']
        end_col = start_col + datain['subsize']
        start_row = datain['start_row']
        end_row = start_row+ datain['subsize']
        lenx = end_row-start_row 
        leny = end_col-start_col  
        pos1x = datain['pos1x'] + start_row
        pos1y = datain['pos1y'] + start_col
        for y in range(leny):
         for x in range(lenx):
             url = urls[y*lenx+ x] 
             image = download_image(url)
             if image is not None :
                 filename = f"{pos1y + y}_{pos1x + x}_tile.jpeg"
                 save_image(image, f"{datain['tiles_dir']}", filename)
                          
        print(json.dumps(datain))
            
        
            
        
            
            
    
           
        
           


