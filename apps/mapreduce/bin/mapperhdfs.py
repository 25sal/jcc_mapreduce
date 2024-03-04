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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Inizializza il client HDFS
hdfs_client = InsecureClient("http://node-master:9870")
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


conta=0
# ---------------------------------------------------------
if __name__ == '__main__':
    logger.warning("mapper starting")
    for line in sys.stdin:
        logger.warning(line)
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
        submatrices_number=datain['submatrices_per_group']
        for y in range(leny):
         for x in range(lenx):
             url = urls[y*lenx+ x]
             # continue to disable downolad
             # continue 
             image = download_image(url)
             if image is not None :
                 filename = f"{pos1y + y}_{pos1x + x}_tile.jpeg"
                 save_image(image, f"{datain['tiles_dir']}", filename)
                          
        conta+=1
        if conta < submatrices_number:                     
          print(json.dumps(datain))     
        else :
          datain['bool']=True
          print(json.dumps(datain))
          conta=0
            
        
            
        
            
            
    
           
        
           


