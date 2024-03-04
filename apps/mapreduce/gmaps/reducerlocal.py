# -*- coding: utf-8 -*-
from PIL import Image
import json
import sys
from io import BytesIO
import cv2
from downloader import getExtent,saveTiff 
import numpy as np
from scale_test import process_image
import os


def merge_tiles(tiles):
    if not tiles:
        return None

    # Estrai le informazioni dalla prima tile per inizializzare il merge
    first_tile = tiles[0]
    start_col = first_tile["start_col"]
    end_col = first_tile["end_col"]
    start_row = first_tile["start_row"]
    end_row = first_tile["end_row"]

    # Calcola le dimensioni dell'immagine risultante
    width = (end_col - start_col) * 256
    height = (end_row - start_row) * 256

    # Crea un'immagine vuota per il merge
    merged_image = Image.new('RGBA', (width, height))

    # Combina le tile nell'immagine risultante
    for tile in tiles:
       col_offset = (tile["pos1y"] - start_col) * 256
       row_offset = (tile["pos1x"] - start_row) * 256
       print(col_offset)
       print(row_offset)

        # Calcola la box per il paste
       box = (
            col_offset, row_offset,
            col_offset + 256, row_offset + 256
        )
        

       merged_image.paste(tile["data"], box)

    return merged_image

def save_merged_image(merged_image, output_path,file_index,image_info):
    try:
        # Assicurati che la cartella di output esista
        if not os.path.isdir(output_path):
          os.makedirs(output_path)
          
    except Exception as e:
        print(f"Errore durante la creazione della directory: {str(e)}")
    
    try:
        if merged_image is not None:
            # Genera un nome di file unico per l'immagine
            filename = f"merged_image_{file_index}.tif"
            try:
             extent = getExtent(image_info["pos1x"], image_info["pos1y"], image_info["pos2x"], image_info["pos2y"], image_info["zoom"], image_info["server"])
             gt = (extent['LT'][0], (extent['RB'][0] - extent['LT'][0]) / merged_image.width, 0, extent['LT'][1], 0,
                 (extent['RB'][1] - extent['LT'][1]) / merged_image.height)
            except Exception as e:
                print(f"errore metodo extent: {str(e)}")
            try:
                merged_image=merged_image.convert('RGB')
            except Exception as e:
                print(f"Errore conver: {str(e)}")
            try:
             r, g, b = cv2.split(np.array(merged_image))
            except Exception as e:
                print(f"Errore cv2: {str(e)}")
            try:
             temp_file_path = os.path.join(output_path, filename)
             saveTiff(r, g, b, gt, temp_file_path)
             

            except Exception as e:
               print(f"Errore tiff: {str(e)}")
            try:
             image, quality = process_image(temp_file_path,temp_file_path)
             
            except Exception as e:
               print(f"Errore process image: {str(e)}")
        else: print("vuota")
    except Exception as e:
        print(f"errore durante la creazione dell'immagine :{str(e)}") 
    try:
        
      
      image.save(f"{output_path}/{filename}",format="JPEG", quality=quality)
    except Exception as e:
        print(f"Errore durante il salvataggio: {str(e)}")
          

current_tiles = []
current_submatrix_info = None
file_index=0;
output_path = "./tmp/output_reducer"  
for line in sys.stdin:
    image_info = json.loads(line)
    image_file = image_info["file_name"]
    image_path = f"./tmp/mapper_output/{image_file}"

    try:
       
       image = Image.open(image_path)
    except Exception as e:
        print(f"Errore durante la creazione dell'immagine:{str(e)}")
        continue # passa alla prossima immagine

    

    if current_submatrix_info is None:
        current_submatrix_info = image_info

    # Verifica se c'è una sovrapposizione orizzontale o verticale con la sottomatrice corrente
    if (
    image_info["start_col"] == current_submatrix_info["start_col"]
    and image_info["end_col"] == current_submatrix_info["end_col"]
    and image_info["start_row"] == current_submatrix_info["start_row"]
    and image_info["end_row"] == current_submatrix_info["end_row"]
    ):
        print("Le immagini sono nella stessa sottomatrice")

        current_tiles.append(
            {
                "start_col": image_info["start_col"],
                "end_col": image_info["end_col"],
                "start_row": image_info["start_row"],
                "end_row": image_info["end_row"],
                "data": image,
                "pos1x":image_info["pos1x"],
                "pos1y": image_info["pos1y"],
                "pos2x": image_info["pos2x"],
                "pos2y":image_info["pos2y"]
            }
        )
    else:
         
         print("Cambio sottomatrice")
        # Effettua il merge delle tile della sottomatrice corrente
        

        # Inizializza una nuova sottomatrice
         current_tiles = [
            {
                "start_col": image_info["start_col"],
                "end_col": image_info["end_col"],
                "start_row": image_info["start_row"],
                "end_row": image_info["end_row"],
                "data": image,
                "pos1x":image_info["pos1x"],
                "pos1y": image_info["pos1y"],
                "pos2x": image_info["pos2x"],
                "pos2y":image_info["pos2y"]
            }
            ]
         print(current_tiles) 
         if current_submatrix_info:
             merged_submatrix = merge_tiles(current_tiles)
             save_merged_image(merged_submatrix, output_path, file_index, current_submatrix_info)
             file_index += 1
            
        
                        
       
         current_submatrix_info = image_info
            
if current_submatrix_info:
    merged_submatrix = merge_tiles(current_tiles)
    save_merged_image(merged_submatrix, output_path, file_index, current_submatrix_info)
