import config
import supervision as sv

from PIL import Image
from pathlib import Path
from rfdetr.util.coco_classes import COCO_CLASSES

def generate_annotated_pictures_and_save(model, seq_num:int, output_folder_name:str):

    seq_dir = config.TRAIN_DATA_DIR / Path(f"seq{seq_num}/thermal")
    
    frames = sorted(seq_dir.glob("*.png"))

    if not frames:
        raise FileNotFoundError(f"No PNG-Pictures found under: {seq_dir}")

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    for i, frame_path in enumerate(frames):
        image = Image.open(frame_path).convert("RGB")
        detections = model.predict(image, threshold=0.5)

        labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

        annotated_image = box_annotator.annotate(image, detections)
        annotated_image = label_annotator.annotate(annotated_image, detections, labels)

        output_path = config.OUTPUT_DIR / Path(f"{output_folder_name}")
        out_path = output_path / f"{i:06d}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        annotated_image.save(out_path)

        if i % 10 == 0:
            print(f"Saved pictures: {i}")

    print(f"Pictures saved under: {output_path.resolve()} \n")

    #TODO: Log hier