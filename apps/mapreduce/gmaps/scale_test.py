import cv2
from io import BytesIO
from PIL import Image
import os

def process_image(input_path,image_path):
    coff_resize = 62.5 / 90
    coff_pix = 9 * 256 / 1500
    
    # Leggi l'immagine originale
    img = cv2.imread(input_path)
    
    # Ridimensiona l'immagine
    img_75 = cv2.resize(img, (1500, 1500), fx=coff_resize, fy=coff_resize)
    
    
    cv2.imwrite(image_path , img_75)
    
    # Calcola la qualità in base al rapporto di dimensioni
    quality = int(95 * (1500 / 1600))
    

    
    # Converti l'immagine in formato PIL
    image_pil = Image.open(image_path)
    
    
    return image_pil, quality

