# Construction Plan Analyzer
**Selected tasks:** detect rooms & extract info    

------------------------------    

## TL;DR
The easier way to run this system is by:
visit [Hello TrueBuilt](http://18.184.182.126) to check if API is working (My EC2 could stop because some maintance or exceeding treshold)

Clone this repository: `https://github.com/mlkonopelski/construction_plan_analyzer.git` or use or own example .pdf 

### Page Info retrieval:
```
curl -X POST -F "file=@data/test/A-492.pdf" -F "find_page_info_method=detection" -H "authorization: Bearer TOKEN" http://18.184.182.126/page_info/
```   
### Rooms detector
```
curl -X POST -F "file=@data/test/A-492.pdf"  -H "authorization: Bearer TOKEN" http://18.184.182.126/rooms/
```   

the token was provided by email to: **Vin**. The inferance can take a bit of time because I used cheap t2.large.


## Aproach 
Because rooms can take any size and shape instead of doing bounding box detection I decided to predict regions and label the training data accoradingly.  
![IMG](readme-resources/true-labels1.png)
![IMG](readme-resources/true-labels2.png)
![IMG](readme-resources/true-labels3.png)

The dataset consisted of 9 PDF files, two of which (`A-192.pdf`, `A-492.pdf`) I used as **out-of-sample** just to verify how well are we doing and to verify API. I chose those to in particular because I thought they best represnt this dataset. 

There was no definition of room therefore I created my own as: "area which is surrounded by walls and at least 1 door." 

While the time was constrained I tried to develop the app to achieve two goals:
1. ML models (test few frameworks) of reasonable accuracy
1. API and DevOps of reasonable advancement 

# Results
The results...

# In-depth ML
## Approaches
### General
Since the panel with all important details of the construction plan (page number, sheet name and Revision) is on the same page as building plan I decided to follow two step approach:
1. Find regions of all rooms
2. Find region of description with information to extract
    1. Cut the region from original plan size: 24k x 18k so the text is easily visible
    2. Run bbox detection of each field of interest. 
    3. Run OCR Tesseract/Easy-OCR on results. 

### Rooms Detection
1. Image Segmentation with YOLOv8 with following specs:
    1. Two classes: Rooms + Description
    1. Architecture: Yolov8 Medium from ultralytics
    1. Original Images resized to 2400x1800
    1. No augmentations what resulted in heavy overfit
1. Same as above but:
    1. Multiple Augmentations: rotation, scale, translation, random flips. While I don't think they represent possible samples in future it should reduce overfitting
    2. Result are not satifactory especially low number of recall of rooms:  
    <img src="https://github.com/mlkonopelski/construction_plan_analyzer/blob/main/readme-resources/yolo-seg-result1.png" width="250"> <img src="https://github.com/mlkonopelski/construction_plan_analyzer/blob/main/readme-resources/yolo-seg-result2.png" width="250">

### OCR "Page Info"
As a easier solution I firstly cut that panel at size common for all training samples. The bbox detection of it was created in case that panel changes position.
1. All 7 training examples were slightly augmented and used for training bounding boxes. I tried YOLOv8 nano, small and medium and surprisingly only medium was giving good results (I anticipated that nano would be suffficient). 
The model gives sufficient results on `A-192` and `A-492` test images:  
<img src="https://github.com/mlkonopelski/construction_plan_analyzer/blob/main/data/predictions/page-info-det/A-192.png" width="200"> <img src="https://github.com/mlkonopelski/construction_plan_analyzer/blob/main/data/predictions/page-info-det/A-492.png" width="200">



# RUN the detection software locally
### Using pre-built docker image
```
docker pull public.ecr.aws/i5l4l1q2/true-built:latest
docker run -d -p 8008:8000 --name true-built-ai public.ecr.aws/i5l4l1q2/true-built:latest

curl -X POST -F "file=@data/test/A-492.pdf" -F "find_page_info_method=detection" -H "authorization: Bearer TOKEN" http://localhost:8008/page_info/
```
### Build from source
```
git clone https://github.com/mlkonopelski/construction_plan_analyzer.git
cd construction_plan_analyzer
docker compose -f docker_compose.yaml up --build --detach

curl -X POST -F "file=@data/test/A-492.pdf" -F "find_page_info_method=detection" -H "authorization: Bearer TOKEN" http://localhost:8008/page_info/
```