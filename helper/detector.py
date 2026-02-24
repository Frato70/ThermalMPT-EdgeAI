import numpy as np
import logging
from typing import Optional

# Das Roboflow RF-DETR Repo muss im Hauptordner liegen
try:
    from rfdetr import RFDETRBase
except ImportError:
    logging.error("❌ rfdetr Modul nicht gefunden. Stelle sicher, dass das Roboflow Repo geklont ist.")
    raise

logger = logging.getLogger(__name__)

class ThermalDetector:
    """
    Wrapper-Klasse für das RF-DETR Modell zur Personenerkennung auf Wärmebildern.
    Implementiert strikt ohne mmyolo oder mmdet.
    """

    def __init__(self, weights_path: str, conf_threshold: float = 0.5):
        """
        Initialisiert den RF-DETR Detektor.
        
        Args:
            weights_path (str): Pfad zu den .pth Gewichten (z.B. Fraunhofer Checkpoint).
            conf_threshold (float): Mindest-Konfidenz für gültige Detektionen.
        """
        self.conf_threshold = conf_threshold
        logger.info(f"Initialisiere RF-DETR Detektor von: {weights_path}")
        
        # Das RFDETRBase Modell handelt das Device (CPU/CUDA) intern
        self.model = RFDETRBase(pretrain_weights=weights_path)
        logger.info("✅ RF-DETR Modell erfolgreich geladen.")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Führt die Objekterkennung auf einem einzelnen Frame durch.
        
        Args:
            frame (np.ndarray): Das Eingabebild (BGR-Format von OpenCV).
            
        Returns:
            np.ndarray: Array der Form (N, 6) mit [x1, y1, x2, y2, score, class_id].
                        Gibt ein leeres Array zurück, wenn nichts gefunden wurde.
        """
        # 1. Inferenz durchführen (gibt ein supervision.Detections Objekt zurück)
        sv_dets = self.model.predict(frame)
        
        # 2. Filtern nach Confidence-Threshold
        mask = sv_dets.confidence >= self.conf_threshold
        sv_dets = sv_dets[mask]

        # 3. Umwandlung in ein sauberes NumPy-Array für den Kalman-Tracker
        if len(sv_dets) > 0:
            dets_array = np.hstack((
                sv_dets.xyxy, 
                sv_dets.confidence[:, np.newaxis], 
                sv_dets.class_id[:, np.newaxis]
            ))
            return dets_array
        else:
            return np.empty((0, 6))