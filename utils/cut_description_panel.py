import cv2
from pdf2image import convert_from_path
import os
import PIL.Image as Image

Image.MAX_IMAGE_PIXELS = None

H = 4000
W = 2000
ROOMS_PATH = os.path.join('data', 'original', 'rooms')
OUTPUT_ROOMS_PATH =  os.path.join('data', 'jpgs', 'rooms', 'descriptions')

pdf_files = os.listdir(ROOMS_PATH)
pdf_files = [f for f in pdf_files if f not in ['A-192.pdf', 'A-492.pdf']] # Exclude completly for Test set

for file_idx, pdf_file in enumerate(pdf_files):
    pages = convert_from_path(os.path.join(ROOMS_PATH, pdf_file), 500)

    for page_idx, page in enumerate(pages):
        
        page_w, page_h = page.size
        crop_x, crop_y = page_w - W, page_h - H
        
        page_cropped = page.crop((crop_x, crop_y, page_w, page_h))
        page_cropped.save(os.path.join(OUTPUT_ROOMS_PATH, f'{file_idx}_{page_idx}.jpg'), 'JPEG')
