import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRNano(pretrain_weights="/home/user/AI-Winterschool-PoseEstimation/rf-detr_thermal_ema.pth", device="cuda", positional_encoding_size=37, patch_size=14)

image = Image.open("https://media.roboflow.com/dog.jpg")
detections = model.predict(image, threshold=0.5)

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.BoxAnnotator().annotate(image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

print(model.model_config.patch_size)