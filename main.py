import os
from typing import Dict, Tuple, Union

import cv2
import easyocr
import numpy as np
from dateutil.parser import parse
from fastapi import (Depends, FastAPI, File, Header, HTTPException, Request,
                     UploadFile)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image
from ultralytics import YOLO

import utils.tools as tools

from utils.authorization import verify_auth
from utils.config import Settings, get_settings
from enum import Enum


class IMG:
    def __init__(self, bytes_string: str) -> None:
        self.pdf = bytes_string
        self.pages = convert_from_bytes(self.pdf)
    
    @staticmethod
    def resize_image(page: Image, target_size: Tuple[int, int] = (2400,  1800)):
        return page.resize(target_size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def cut_page_info(page: Image, WH:Tuple[int]=(2000, 4000), WH_RATIO: Union[Tuple[int], None]=None) -> Image:

        # PFD comes with different size from curl than original. #TODO: Explore it
        # page = page.resize((2400,  1800), Image.Resampling.LANCZOS)
        # w_ratio, h_ratio is typical ratio for panel
        # h_ratio = 4000 / 18000
        # w_ratio = 2000 / 24000
        # crop_x, crop_y = page_w - int(page_w*w_ratio), page_h - int(page_h*h_ratio)
        
        page_w, page_h = page.size
        if WH:
            W, H = WH
            crop_x, crop_y = page_w - W, page_h - H
        elif WH_RATIO:
            W_RATIO, H_RATIO = WH_RATIO
            crop_x, crop_y = page_w - int(page_w*W_RATIO), page_h - int(page_h*H_RATIO)
            
        page = page.crop((crop_x, crop_y, page_w, page_h))

        return page


class Detection:
    
    def __init__(self, algorithm: str) -> None:
        
        self.algorithm = algorithm
        self._init_model()
        
    def _init_model(self):
        assert self.algorithm in ['yolov8m'], "Models implemented for now: yolov8m"
        
        if self.algorithm == 'yolov8m':
            self.model = YOLO(os.path.join('.models', 'yolov8m-seg.pt'))
    
    
    def cut_page_info(self, img: Image):
        
        W, H = img.size
        result = self.model(img)[0]
        boxes_count = result.boxes.shape[0]
        best_bbox_score = 0
        crop_bbox = None
        for bbox_id in range(boxes_count):
            label = result.boxes.cls[bbox_id].item()
            conf = result.boxes.conf[bbox_id].item()
            if label==0 and conf> best_bbox_score:
                crop_bbox = result.boxes.xyxy[bbox_id].cpu().tolist()
                # x1_n, y1_n, x2_n, y2_n = result.boxes.xyxyn[bbox_id].cpu().tolist()
                best_bbox_score = conf
        if crop_bbox:
            # x1, y1, x2, y2 = x1_n*W, y1_n*H, x2_n*W, y2_n*H
            img_cropped = img.crop((crop_bbox[0] - 10, crop_bbox[1] - 10, W, H))
        else:
            # in case detection algorithm didn't find any bbox
            img_cropped = IMG.cut_page_info(img)
        
        return img_cropped
            
    
    def run(self, img: Image, orig_size: Tuple[int, int]) -> None:
            
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
        
    def format_to_json(self, file_name: str) -> Dict:
        
        json = {
            "type": "rooms",
            "imageId": file_name,
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
        
        if not len(revision.keys()) == 3:
            # print(f'Missing 1 or more keys from revision: {revision}')
            revision = {}
        return revision

    def run(self, img: Image) -> None:

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
                        sheet_number_conf = conf
                    elif names[cls_] == 'sheet_name' and conf > sheet_name_conf:
                        img_to_ocr = img.crop((box[0], box[1], box[2], box[3]))
                        page_info['sheet_name'] = self._ocr_img(img_to_ocr)
                        sheet_name_conf = conf
                    elif names[cls_] == 'revision':
                        img_to_ocr = img.crop((box[0], box[1], box[2], box[3]))
                        revision = self._ocr_revisions(img_to_ocr)
                        if revision and revision not in revisions: # In case revision is empty bbox or there are 2 bouding boxes on the same text
                            revisions.append(revision)
            
            page_info['revision'] = revisions                      
        else:
            print('Found no prediction for boxes')
            
        self.page_info = page_info
        
    def format_to_json(self, file_name: str) -> Dict:
        json = {"imageId": file_name}
        json.update(self.page_info)
        return json


#---------------------------------------------
#           RUN APP
#---------------------------------------------

# API
app = FastAPI(title='TrueBuilt Helper`', 
              description='API for Construction Plans', 
              version='1.0', 
              debug=True)


@app.on_event('startup')
async def load_models():
    tools.detection = Detection('yolov8m')
    tools.info_extraction = InfoExtraction('yolov8m')


# TODO: Create Home screen with API description
@app.get("/")
def read_root():
    return {"Hello": "TrueBuilt"}


@app.post('/rooms/') # http POST
async def rooms_view(file: UploadFile = File(...),
                     authorization = Header(None),
                     settings: Settings() = Depends(get_settings)
                    ):
    verify_auth(authorization, settings)

    bytes_str = await file.read()

    try:
        pages = IMG(bytes_str).pages
        for page in pages:
            page_resized = IMG.resize_image(page)
            tools.detection.run(page_resized, page.size)
            r_json = tools.detection.format_to_json(file.filename)

    except Exception as e:
        raise HTTPException(detail=f'{e}', 
                            status_code=400)
    
    return r_json

    
class PageInfoMethod(str, Enum):
    fix_crop = 'fix_crop'
    detection = 'detection'

@app.post('/page_info/') # http POST
async def page_info_view(file: UploadFile = File(...),
                         find_page_info_method: PageInfoMethod = 'detection',
                         authorization = Header(None),
                         settings: Settings() = Depends(get_settings)
                          ):

    verify_auth(authorization, settings)
    assert find_page_info_method in ['fix_crop', 'detection'], 'Only possible methods: ["fix_crop", "detection"]'

    bytes_str = await file.read()

    try:
        pages = IMG(bytes_str).pages
        for page in pages:
            if find_page_info_method == 'fix_crop':
                page_cropped = IMG.cut_page_info(page)
            elif find_page_info_method == 'detection':
                page_cropped = tools.detection.cut_page_info(page)
            tools.info_extraction.run(page_cropped)
            r_json = tools.info_extraction.format_to_json(file.filename)

    except Exception as e:
        raise HTTPException(detail=f'{e}', status_code=400)

    return r_json


if __name__ == '__main__':
    

    Image.MAX_IMAGE_PIXELS = None

    IMG_SIZE = (2400,  1800)

    FILE = 'A-192.pdf' # A-192.pdf, A-492.pdf
    TEST_PATH = os.path.join('data', 'test')
    
    # FILE = '_13.pdf'
    # TEST_PATH = 'data/original/rooms'

    pages = convert_from_path(os.path.join(TEST_PATH, FILE), 500)
    
    # FIND ROOMS
    page_resized = IMG.resize_image(pages[0])
    detection = Detection('yolov8m')
    detection.run(page_resized, pages[0].size)
    # print(detection.format_to_json(FILE))
    
    # FIND PAGE INFO
    # page_cropped = IMG.cut_page_info(pages[0])
    page_cropped = detection.cut_page_info(pages[0])
    info_extraction = InfoExtraction('yolov8m')
    info_extraction.run(page_cropped)
    print(info_extraction.format_to_json(FILE))
