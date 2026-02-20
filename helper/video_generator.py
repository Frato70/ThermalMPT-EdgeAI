import config
import cv2
import numpy as np

from pathlib import Path
from PIL import Image

def generate_video_from_images(output_name: str, images_folder_name: str, fps: int = 20):
    image_dir = config.OUTPUT_DIR / Path(images_folder_name)
    frames = sorted(image_dir.glob("*.png"))

    if not frames:
        raise FileNotFoundError(f"No PNG-Pictures found under: {image_dir}")
    
    output_path = config.OUTPUT_DIR / Path(f"{output_name}.mp4")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_writer = None

    for frame_path in frames:
        # Bild laden
        pil_img = Image.open(frame_path).convert("RGB")
        frame_rgb = np.array(pil_img)

        # OpenCV erwartet BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # VideoWriter beim ersten Frame initialisieren
        if video_writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

            if not video_writer.isOpened():
                raise RuntimeError("VideoWriter konnte nicht geöffnet werden.")

        video_writer.write(frame_bgr)

    video_writer.release()
    print(f"Video saved under: {output_path.resolve()} \n")

    #TODO: Log hier