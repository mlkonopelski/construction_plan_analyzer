from pdf2image import convert_from_path
import os
import PIL.Image as Image

Image.MAX_IMAGE_PIXELS = None

IMG_SIZE = (2400,  1800)

ROOMS_PATH = os.path.join('data', 'original', 'rooms')
OUTPUT_ROOMS_PATH =  os.path.join('data', 'jpgs', 'rooms', '2400_1800')

pdf_files = os.listdir(ROOMS_PATH)
pdf_files = [f for f in pdf_files if f not in ['A-192.pdf', 'A-492.pdf']] # Exclude completly for Test set

for file_idx, pdf_file in enumerate(pdf_files):
    pages = convert_from_path(os.path.join(ROOMS_PATH, pdf_file), 500)

    for page_idx, page in enumerate(pages):
        print(f'{file_idx}_{page_idx} size: {page.size}')
        # page_resized = cv2.resize(page, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        page_resized = page.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        print(f'{file_idx}_{page_idx} size: {page_resized.size}')
        page_resized.save(os.path.join(OUTPUT_ROOMS_PATH, f'{file_idx}_{page_idx}.jpg'), 'JPEG')
