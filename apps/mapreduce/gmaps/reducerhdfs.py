import cv2
from PIL import Image
import json
import sys
from io import BytesIO
import cv2
from hdfs import InsecureClient
import os
import shutil

directory_path = "/percorso/della/tua/directory_da_rimuovere"

try:
    shutil.rmtree(directory_path)
    print(f"La directory {directory_path} e tutti i suoi contenuti sono stati rimossi con successo.")
except OSError as e:
    print(f"Errore durante la rimozione della directory e dei suoi contenuti: {str(e)}")

hdfs_client = InsecureClient("http://node-master:50070")
hdfs_path="/immagini/"
hdfs_client.makedirs(hdfs_path, permission=755)

def savehdfs(immagine,hdfs_path,pos1x,pos1y,lenx):
     try:
        
        filename=f"{pos1x}_{pos1y}_{lenx}_1500.tiff"
        if immagine is not None:
            
            with hdfs_client.write(f"{hdfs_path}/{filename}", overwrite=True) as writer:
               immagine.save(writer,format="TIFF")
         
        else: print("vuota")

     except Exception as e:
         # Gestisci gli errori durante la scrittura su HDFS
       print(f"Errore durante la scrittura dell'immagine su HDFS: {str(e)}")
    



def resize(infile: str, outfile: str):
    coff_resize = 62.5/90
    coff_pix = 9*256/1500
    # if this image is 2304 px 
    img=cv2.imread(infile)
    # this should be 1600px 1m per px
    img_75 = cv2.resize(img, (1500,1500), fx=coff_resize,fy=coff_resize)
    cv2.imwrite(outfile, img_75) 

def merge_tiles(pos1x, pos1y, lenx, leny, tiles_dir):
    width = lenx * 256
    height = leny * 256
    # Crea un'immagine vuota per il merge
    merged_image = Image.new('RGBA', (width, height))
    # Combina le tile nell'immagine risultante
    for y in range(leny):
        for x in range(lenx):
            with hdfs_client.read(f"{tiles_dir}/{pos1y+y}_{pos1x+x}_tile.jpeg") as reader:
                small_pic = Image.open(BytesIO(reader.read()))  # Corretta apertura dell'immagine
                merged_image.paste(small_pic, (x * 256, y * 256))
    print('Tiles merge completed')
    
    return merged_image

output_dir = "/home/spark/immagini/"

if __name__ == "__main__":
    for line in sys.stdin: 
        json_str =json.loads(line)
        # Estrai le informazioni dalla prima tile per inizializzare il merge
        start_col = json_str["start_col"]
        end_col = start_col+json_str["subsize"]
        start_row = json_str["start_row"]
        end_row = start_row+json_str["subsize"]
        lenx = end_row-start_row
        leny = end_col-start_col
        pos1x = json_str['pos1x'] + start_row
        pos1y = json_str['pos1y'] + start_col
        outpic = merge_tiles(pos1x, pos1y, lenx, leny, json_str['tiles_dir'])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        outpic = outpic.convert('RGB')
        outpic.save(f"{output_dir}/{pos1x}_{pos1y}_{lenx}.tiff", format="TIFF")
        resize(f"{output_dir}/{pos1x}_{pos1y}_{lenx}.tiff",f"{output_dir}/{pos1x}_{pos1y}_{lenx}_1500.tiff")
        immagine=Image.open(f"{output_dir}/{pos1x}_{pos1y}_{lenx}_1500.tiff")
        savehdfs(immagine,hdfs_path,pos1x,pos1y,lenx)
        try:
            shutil.rmtree(directory_path)
            print(f"La directory {directory_path} e tutti i suoi contenuti sono stati rimossi con successo.")
        except OSError as e:
            print(f"Errore durante la rimozione della directory e dei suoi contenuti: {str(e)}")
       
       