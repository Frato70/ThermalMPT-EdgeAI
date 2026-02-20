import config
import core.online_picture_annotator as opg
import helper.video_generator as vg

from rfdetr import RFDETRBase
from pathlib import Path

model = RFDETRBase(pretrain_weights=config.PRETRAINED_WEIGHTS, device=config.USED_DEVICE)

opg.generate_annotated_pictures_and_save(model=model, seq_num=26, output_folder_name="output_pictures_1")
vg.generate_video_from_images(output_name="video_output", images_folder_name="output_pictures_1")
