import cv2
import torch
import numpy as np
import logging
import gc
from typing import Dict, List, Any, Tuple

try:
    from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
except ImportError:
    # Manueller Fallback, falls die __init__.py von transformers fehlerhaft ist
    from transformers.models.vitpose.modeling_vitpose import VitPoseForPoseEstimation
    from transformers.models.vitpose.image_processing_vitpose import VitPoseImageProcessor

# Logger-Setup für sauberes Konsolen-Feedback
logger = logging.getLogger(__name__)

class ThermalPoseEngine:
    """
    Klasse für die Inferenz der menschlichen Körperhaltung auf Wärmebildern.
    Nutzt das ViTPose-Modell via Hugging Face Transformers.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        logger.info(f"Initialisiere ThermalPoseEngine auf Device: {self.device}")
        
        try:
            base_model_name = "usyd-community/vitpose-base-simple"
            
            self.processor = VitPoseImageProcessor.from_pretrained(base_model_name)
            self.model = VitPoseForPoseEstimation.from_pretrained(base_model_name)
            
            if model_path.endswith('.pth'):
                logger.info(f"Lade Thermal-Gewichte von: {model_path} (in CPU-RAM)")
                state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
                self.model.load_state_dict(state_dict, strict=False)
                
                del state_dict
                gc.collect()
                torch.cuda.empty_cache()
            
            # FP16 Diät NUR anwenden, wenn wir auf CUDA sind!
            if self.device == "cuda":
                logger.info("Konvertiere ViTPose zu FP16 für CUDA...")
                self.model = self.model.half()
            
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info(f"Verschiebe Modell auf {self.device}...")
            self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("✅ ViTPose Modell erfolgreich geladen und bereit.")
            
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden von ViTPose: {e}")
            raise

    def estimate_poses(self, frame: np.ndarray, active_tracks: Dict[int, np.ndarray]) -> Dict[int, List[Tuple[float, float, float]]]:
        poses_result = {}
        
        if not active_tracks:
            return poses_result

        img_h, img_w = frame.shape[:2]

        for track_id, bbox in active_tracks.items():
            x1, y1, x2, y2 = map(int, bbox[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            person_crop = frame[y1:y2, x1:x2]
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            crop_h, crop_w = person_rgb.shape[:2]
            
            # FIX: HF braucht ZWINGEND eine lokale Box für das zugeschnittene Eingabebild!
            local_box = [[[0, 0, int(crop_w), int(crop_h)]]]
            inputs = self.processor(person_rgb, boxes=local_box, return_tensors="pt")
            
            pixel_values = inputs.pixel_values.to(self.device)
            if self.device == "cuda":
                pixel_values = pixel_values.half()

            with torch.no_grad():
                outputs = self.model(pixel_values)

            outputs.heatmaps = outputs.heatmaps.float()
            
            # 2. Genialer Trick: Post-Processing mit der GLOBALEN Box
            global_box = [[[x1, y1, x2, y2]]]
            raw_results = self.processor.post_process_pose_estimation(
                outputs, 
                boxes=global_box
            )[0]
            
            if isinstance(raw_results, list) and len(raw_results) > 0:
                pose_dict = raw_results[0]
            elif isinstance(raw_results, dict):
                pose_dict = raw_results
            else:
                pose_dict = {}
            
            global_keypoints = []
            keypoints = pose_dict.get("keypoints", [])
            scores = pose_dict.get("scores", [])
            
            for kp, score in zip(keypoints, scores):
                global_x, global_y = kp[0].item(), kp[1].item()
                conf = score.item()
                global_keypoints.append((global_x, global_y, conf))
                
            poses_result[track_id] = global_keypoints

        return poses_result