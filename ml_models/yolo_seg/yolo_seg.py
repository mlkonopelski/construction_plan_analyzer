from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m-seg.pt")  # load a pretrained model (recommended for training)

model.train(data="/Users/mkonopelski/Projects/research/analyze_construction_plans/data/yolo-ds/dataset.yaml",
            cfg='/Users/mkonopelski/Projects/research/analyze_construction_plans/ml-models/yolo-seg/yolo-seg-fullaug.yaml',
            name='yolov8-seg-fullaug')


class YOLOSegmentation:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    def __call__(self, img):
        return self.model(img)