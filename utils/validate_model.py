from pdf2image import convert_from_path
import os
import PIL.Image as Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
import os
import datetime 

# ---------------------------------------
# READ IMAGE 
Image.MAX_IMAGE_PIXELS = None

IMG_SIZE = (2400,  1800)

# FILE = 'A-492.pdf' # A-192.pdf, A-492.pdf
# TEST_PATH = os.path.join('data', 'test')
# FILE = '_13.pdf'
FILE = 'A-191.pdf'
TEST_PATH = 'data/original/rooms'

pages = convert_from_path(os.path.join(TEST_PATH, FILE), 500)
time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M')

# ---------------------------------------
# VISUALIZATION: SEGMENTATION WITH YOLO

for page_idx, page in enumerate(pages):
    page_resized = page.resize(IMG_SIZE, Image.Resampling.LANCZOS)
# page_resized = page.copy()

# model_path = 'ml_models/yolo_seg/u_yolo_artifacts'
# model = YOLO(model_path + '/yolov8-seg-fullaug/weights/best.pt')
model_path = '.models/yolov8m-seg.pt'
model = YOLO(model_path)

result = model(page_resized)[0]

names = result.names
# Visualize masks
if result:
    plt.imshow(page_resized, cmap='gray') # I would add interpolation='none'
    masks_count = result.masks.shape[0]
    for mask_id in range(masks_count):
        bbox = result.boxes.xywh[mask_id]
        label = result.boxes.cls[mask_id].item()
        conf = result.boxes.conf[mask_id].item()
        
        if conf >= 0.35:
            mask = result.masks.data[mask_id][masks_count:,:]

            if label == 0 and conf > 0.9:
                plt.imshow(mask, cmap='GnBu', alpha=0.5*(mask>0))
            else:
                plt.imshow(mask, cmap='PuRd', alpha=0.5*(mask>0))
            
            plt.text(bbox[0], bbox[1], f'{round(conf, 2)}: {names[label]} ', fontsize=6)
else:
    print('NO RESULTS :(')

print('waiting...')

# ---------------------------------------
# VISUALIZATION: BBOX DETECTION WITH YOLO

H = 4000
W = 2000

page = pages[0]
    
page_w, page_h = page.size
crop_x, crop_y = page_w - W, page_h - H
page_cropped = page.crop((crop_x, crop_y, page_w, page_h))
cv2_image =  cv2.cvtColor(np.array(page_cropped), cv2.COLOR_RGB2BGR)

# model_path = 'ml_models/yolo_det/u_yolo_artifacts/yolov8n-det-littleaug/weights/best.pt'
model_path = '.models/yolov8m-det.pt'
# model_path = 'ml_models/yolo_seg_nano/u_yolo_artifacts/yolov8n-seg-fullaug/weights/best.pt'
model =  YOLO(model_path)
names = model.names
# page_cropped = cv2.imread('page_cropped.jpg')
results = model(page_cropped)
boxes = results[0].boxes.xyxy.cpu().tolist()
clss = results[0].boxes.cls.cpu().tolist()
confs = results[0].boxes.conf.cpu().tolist()
annotator = Annotator(cv2_image, line_width=2, example=names)
# annotator = Annotator(page_cropped, line_width=2, example=names)

idx = 0
if boxes is not None:
    for box, cls_, conf in zip(boxes, clss, confs):
        if conf >= 0.5:
            idx += 1
            annotator.box_label(box, color=colors(int(cls_), True), label=names[int(cls_)])
        # crop_obj = cv2_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        # cv2.imwrite(os.path.join('data/predictions', str(idx)+".png"), crop_obj)
    # cv2.imshow("ultralytics", cv2_image)
    cv2.imwrite(os.path.join('data/predictions', FILE.replace('.pdf', '') + time_stamp + ".png"), cv2_image)
    # cv2.imwrite(os.path.join('data/predictions', FILE.replace('.pdf', '') +".png"), page_cropped)
else:
    print('Found no prediction for boxes')        

# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break
print('waiting...')

# Helper methods to visualize masks
# def draw_mask(image, mask_generated) :
#   masked_image = image.copy()
  
#   mask_generated = mask_generated.numpy().transpose((1,2,0)).astype(int)
  
#   masked_image = np.where(mask_generated,
#                           np.array([0,255,0], dtype='uint8'),
#                           masked_image)

#   masked_image = masked_image.astype(np.uint8)

#   return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)


# segmented_image = draw_mask(page_resized, result.masks.data)
# cv2.imshow(segmented_image)
# cv2.waitKey(0)

# Suprisingly supervision doesn't work with segmentation masks
# Validation helper
# import supervision as sv
# bounding_box_annotator = sv.BoundingBoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# labels = [
#     model.model.names[class_id]
#     for class_id
#     in detections.class_id
# ]

# annotated_image = bounding_box_annotator.annotate(
#     scene=page_resized, detections=detections)
# annotated_image = label_annotator.annotate(
#     scene=annotated_image, detections=detections, labels=labels)