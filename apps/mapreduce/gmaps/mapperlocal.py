# -*- coding: utf-8 -*-

from http import server
import logging
import time
import json
import sys
from downloader import wgs_to_tile, get_url
import requests
from PIL import Image
from io import BytesIO
import os


logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)

def download_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Converti il contenuto dell'immagine in un oggetto Image
            image = Image.open(BytesIO(response.content))
            return image
        else:
            # Gestisci il caso in cui la richiesta non abbia successo
            print(f"Errore nel download dell'immagine da URL: {url}")
            return None
    except Exception as e:
        # Gestisci eventuali eccezioni nel download dell'immagine
        print(f"Errore nel download dell'immagine: {str(e)}")
        return None

def get_urls(line):
    datain = json.loads(line)
    server = datain['server']
    start_col = datain['pos1y'] + datain['start_col']
    end_col = start_col + datain['subsize']
    start_row = datain['pos1x'] + datain['start_row']
    end_row = start_row + datain['subsize']
    urls = []
    
    for j in range(start_col, end_col):
        for i in range(start_row, end_row):
            dynamic_pos1x = datain['pos1x'] + (i - start_row)
            dynamic_pos1y = datain['pos1y'] + (j - start_col)
            dynamic_pos2x = datain['pos2x'] + (i - start_row)
            dynamic_pos2y = datain['pos2y'] + (j - start_col)

            url = get_url(datain['server'], i, j, datain['zoom'], datain['style'])
            
            urls.append((url, dynamic_pos1x, dynamic_pos1y, dynamic_pos2x, dynamic_pos2y))
            
    return urls, start_col, end_col, start_row, end_row, datain['zoom'], server


def save_image_and_info(image,image_info, output_dir):
    if not os.path.isdir(output_dir):
          os.makedirs(output_dir)
    if image is not None:
        # Genera un nome di file unico per l'immagine
        timestamp = str(int(time.time()))
        filename = f"image_{timestamp}.jpeg"

        # Salva l'immagine nella directory temporanea 
        image.save(f"{output_dir}/{filename}", format="JPEG")
        
       


        # Aggiungi le informazioni necessarie per il merging
        image_info["file_name"] = filename
        return image_info

if __name__ == '__main__':
    

    # Crea una directory temporanea su HDFS per il Mapper
    temp_dir = "./tmp/mapper_output/"
  

    for line in sys.stdin:
        urls, start_col, end_col, start_row, end_row, zoom, server = get_urls(line)
        for url, pos1x, pos1y, pos2x, pos2y in urls:
            immagine = download_image(url)
            if immagine is not None:
                image_info = {
                    "start_col": start_col,
                    "end_col": end_col,
                    "start_row": start_row,
                    "end_row": end_row,
                    "pos1x": pos1x,
                    "pos1y": pos1y,
                    "pos2x": pos2x,
                    "pos2y": pos2y,
                    "zoom": zoom,
                    "server": server
                }

                save_image_and_info(immagine, image_info, temp_dir)

                # Stampa il dizionario come stringa JSON nello stdout
                print(json.dumps(image_info))