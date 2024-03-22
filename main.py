import os
from typing import Dict, Tuple

import cv2
import easyocr
import numpy as np
from dateutil.parser import parse
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image
from ultralytics import YOLO


class IMG:
    def __init__(self, bytes_string: str) -> None:
        self.pdf = bytes_string
        self.pages = convert_from_bytes(self.pdf)
    
    @staticmethod
    def resize_image(page: Image, target_size: Tuple[int, int] = (2400,  1800)):
        return page.resize(target_size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def cut_page_info(page: Image) -> Image:
        H = 4000
        W = 2000
        page_w, page_h = page.size
        crop_x, crop_y = page_w - W, page_h - H
        return page.crop((crop_x, crop_y, page_w, page_h))


class Detection:
    
    def __init__(self, algorithm: str) -> None:
        
        self.algorithm = algorithm
        self._init_model()
        
    def _init_model(self):
        assert self.algorithm in ['yolov8m'], "Models implemented for now: yolov8m"
        
        if self.algorithm == 'yolov8m':
            self.model = YOLO(os.path.join('.models', 'yolov8m-seg.pt'))
    
    
    def run(self, identifier: str, img: Image, orig_size: Tuple[int, int]) -> None:
        
        self.identifier = identifier
        
        result = self.model(img)[0]
        
        det = []

        room_id = 0
        masks_count = result.masks.shape[0]
        for mask_id in range(masks_count):
            label = result.boxes.cls[mask_id].item()
            conf = result.boxes.conf[mask_id].item()
            
            if label==1 and conf >= 0.5:
                mask_norm = result.masks.xyn[mask_id]
                mask_array = mask_norm * orig_size          
                mask = [{'x': int(round(tup[0])), 'y': int(round(tup[1]))} for tup in mask_array]
                det.append({
                'roomId': f'room_{room_id}',
                'vertices': mask,
                'confidence': round(conf, 2)})
                room_id += 1
            
        self.det = det
        
    def format_to_json(self) -> Dict:
        
        json = {
            "type": "rooms",
            "imageId": self.identifier,
            "detectionResults": {
                "rooms": self.det
                    }
        }
        return json


class InfoExtraction:
    def __init__(self, algorithm: str = 'yolov8m') -> None:
        self.algorithm = algorithm
        self._init_model()
        self.reader = easyocr.Reader(['en'], gpu=False)
        
    def _init_model(self):
        assert self.algorithm in ['yolov8m'], "Models implemented for now: yolov8m"
        if self.algorithm == 'yolov8m':
            self.model = YOLO(os.path.join('.models', 'yolov8m-det.pt'))
            
    def _ocr_img(self, img: Image) -> str:
        
        result_str = ''        

        gray = cv2.cvtColor(np.array(img) , cv2.COLOR_RGB2GRAY)
        results = self.reader.readtext(gray)
        for result in results:
            if result[2] > 0.5:
                result_str += ' ' + result[1]
    
        return result_str.lstrip()  # remove empty space in case result is just one word
    
    def _ocr_revisions(self, img: Image) -> str:

        revision = {}
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        results = self.reader.readtext(gray)
        for result in results:
            if result[1].isdigit() and result[2] > 0.5: # MAke sure we happy with OCR
                revision['number'] = int(result[1])
                continue
            try:
                date = parse(result[1], fuzzy=False)
                revision['date'] = str(date.date())
                continue
            except ValueError:
                'do nothing'
            # We assume that If no conditions above are true the string is description
            revision['description'] = result[1]
            
        return revision
                
        
    def run(self, identifier: str, img: Image) -> None:
        self.identifier = identifier
        
        names = self.model.names
        result = self.model(img)[0]
        
        boxes = result.boxes.xyxy.cpu().tolist()
        clss = result.boxes.cls.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        
        page_info = {}
        revisions = []
        sheet_name_conf = 0
        sheet_number_conf = 0

        if boxes is not None:
            for box, cls_, conf in zip(boxes, clss, confs):
                if conf >= 0.5:
                    if names[cls_] == 'sheet_number' and conf > sheet_number_conf: # Just to make sure we take bbox with largest conf    
                        img_to_ocr = img.crop((box[0], box[1], box[2], box[3]))
                        page_info['sheet_number'] = self._ocr_img(img_to_ocr)
                    elif names[cls_] == 'sheet_name' and conf > sheet_name_conf:
                        img_to_ocr = img.crop((box[0], box[1], box[2], box[3]))
                        page_info['sheet_name'] = self._ocr_img(img_to_ocr)
                    elif names[cls_] == 'revision':
                        img_to_ocr = img.crop((box[0], box[1], box[2], box[3]))
                        revision = self._ocr_revisions(img_to_ocr)
                        if revision: # In case revision is empty bbox
                            revisions.append(revision)
            
            page_info['revision'] = revisions                      
        else:
            print('Found no prediction for boxes')
            
        self.page_info = page_info
        
    def format_to_json(self):
        return self.page_info


if __name__ == '__main__':
    

    Image.MAX_IMAGE_PIXELS = None

    IMG_SIZE = (2400,  1800)

    FILE = 'A-492.pdf' # A-192.pdf, A-492.pdf
    TEST_PATH = os.path.join('data', 'test')

    pages = convert_from_path(os.path.join(TEST_PATH, FILE), 500)
    page_resized = IMG.resize_image(pages[0])
    detection = Detection('yolov8m')
    detection.run('test', page_resized, pages[0].size)
    print(detection.format_to_json())
    
    page_cropped = IMG.cut_page_info(pages[0])
    info_extraction = InfoExtraction('yolov8m')
    info_extraction.run('test', page_cropped)
    print(info_extraction.format_to_json())