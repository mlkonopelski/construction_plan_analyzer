# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, FastRCNNPredictor, maskrcnn_resnet50_fpn
import cv2
import numpy as np
import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, maskrcnn_resnet50_fpn


FEATURES = 2  # I'm building this model just for Rooms prediction
DEVICE = 'cpu'  # TODO: check support for mps
IMG_SIZE=[600,600]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, mode: str):
        assert mode in ['train', 'val']
        self.imgs_path = f'data/yolo-ds/{mode}/images/'
        self.imgs = sorted(os.listdir(self.imgs_path))
        self.regions_path = f'data/yolo-ds/{mode}/labels/'
        self.regions = sorted(os.listdir(self.regions_path))
        
    def _yolomask2torchmask(self, region_path, shape):
        h, w, c = shape
        mask = np.zeros((w, h), dtype=np.uint8)
        with open(region_path, 'r') as f:
            labels = f.readlines()

        coordinates = []
        boxes = []
        im_mask = []
        categories = []
        for label in labels:
            label = label.rstrip().split(' ')
            cat = label[0]
            # if int(cat) == 1:
            coordinates_array = label[1:]
            coordinates_temp = []
            for cord_ix in range(0, len(coordinates_array), 2):
                coordinates_temp.append((int(float(coordinates_array[cord_ix])*w), 
                                        int(float(coordinates_array[cord_ix+1])*h)))
            coordinates_temp_array = np.array(coordinates_temp)
            x1, y1 = min(coordinates_temp_array[:, 0]), min(coordinates_temp_array[:, 1])
            x2, y2 = max(coordinates_temp_array[:, 0]), max(coordinates_temp_array[:, 1])
            boxes.append([x1, y1, x2, y2])
            coordinates.append(coordinates_temp_array)
            mask_temp = cv2.fillPoly(mask, coordinates, 1)
            im_mask.append(mask_temp)
            categories.append(int(cat))
        im_mask = np.stack(im_mask, axis=0)
        return categories, boxes, im_mask

        
    def __getitem__(self , idx):
        img = cv2.imread(self.imgs_path+ self.imgs[idx])
        img = cv2.resize(img, IMG_SIZE, cv2.INTER_LINEAR)
        categories, boxes, mask = self._yolomask2torchmask(self.regions_path + self.regions[idx],
                                               img.shape)        
        
        num_objs = len(boxes)
        
        boxes = torch.as_tensor(boxes , dtype = torch.float32)
        labels = torch.ones((num_objs, ), dtype = torch.int64)
        mask = torch.as_tensor(mask , dtype = torch.uint8)
        img = torch.as_tensor(img.transpose(2, 1, 0), dtype=torch.float32)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = mask
        return img, target
    
    def __len__(self):
        return len(self.imgs)

def custom_collate(data):
  return data

def train():
    
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features , FEATURES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask , hidden_layer , FEATURES)
    
    model = model.to(DEVICE)
    
    train_dl = torch.utils.data.DataLoader(CustomDataset('train'), 
                                           batch_size = 2, 
                                           shuffle = True, 
                                           collate_fn = custom_collate,
                                           num_workers = 1 , 
                                           pin_memory = False)
    val_dl = torch.utils.data.DataLoader(CustomDataset('val'),
                                         batch_size = 2,
                                         shuffle = True,
                                         collate_fn = custom_collate, 
                                         num_workers = 1, 
                                         pin_memory = False)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    all_train_losses = []
    all_val_losses = []
    print_init = False
    for epoch in range(5):
        train_epoch_loss = 0
        val_epoch_loss = 0
        model.train()
        for i , dt in enumerate(train_dl):
            imgs = [dt[0][0].to(DEVICE) , dt[1][0].to(DEVICE)]
            targ = [dt[0][1] , dt[1][1]]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targ]
            optimizer.zero_grad()
            loss = model(imgs , targets)
            if not print_init:
                print(f'-------INIT LOSS-------\n{loss}\n----------------')
                print_init = True
            losses = sum([l for l in loss.values()])
            train_epoch_loss += losses.cpu().detach().numpy()
            losses.backward()
            optimizer.step()
        all_train_losses.append(train_epoch_loss)
        print(loss)
        # with torch.no_grad():
        #     for j , dt in enumerate(val_dl):
        #         imgs = [dt[0][0].to(DEVICE) , dt[1][0].to(DEVICE)]
        #         targ = [dt[0][1] , dt[1][1]]
        #         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targ]
        #         loss = model(imgs , targets)
        #         losses = sum([l for l in loss.values()])
        #         val_epoch_loss += losses.cpu().detach().numpy()
        #     all_val_losses.append(val_epoch_loss)
        # print(epoch , "  " , train_epoch_loss , "  " , val_epoch_loss)
        
    return model

if __name__ == '__main__':
    
    model = train()

    test_img = cv2.imread('data/yolo-ds/val/images/0_0_jpg.rf.391a9f6932cc4ab81a734762e11bbb18.jpg')
    test_img = cv2.resize(test_img, IMG_SIZE, cv2.INTER_LINEAR)
    test_img = torch.as_tensor(test_img.transpose(2, 1, 0), dtype=torch.float32)

    with torch.no_grad():
        model.val()
        results = model.predict(test_img)

    results = ...