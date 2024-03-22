from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

model.train(data="/Users/mkonopelski/Projects/research/analyze_construction_plans/data/yolo-det-ds/dataset.yaml",
            cfg='/Users/mkonopelski/Projects/research/analyze_construction_plans/ml_models/yolo_det/yolo_det.yaml',
            name='yolov8n-det-littleaug')

class YOLOSegmentation:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    def __call__(self, img):
        return self.model(img)