import cv2
import time
import json
import logging
import argparse
import sys
import os
import numpy as np


# --- 🚨 ENGINEER'S BYPASS FÜR DEPENDENCY HELL 🚨 ---
import sys
import os
import accelerate.utils.memory
import transformers.utils
import torch

# Zwinge Python, zuerst im User-Verzeichnis nach Paketen zu suchen
user_site = os.path.expanduser("~/.local/lib/python3.10/site-packages")
if user_site in sys.path:
    sys.path.remove(user_site)
sys.path.insert(0, user_site)

import accelerate.utils.memory
import transformers.utils
import torch

# COCO Skelett-Verbindungen (Index-Paare der 17 Keypoints)
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), # Beine & Hüfte
    (5, 11), (6, 12), (5, 6),                         # Torso
    (5, 7), (7, 9), (6, 8), (8, 10),                  # Arme
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),           # Gesicht
    (3, 5), (4, 6)                                    # Ohren zu Schultern
]

# Fix für accelerate/peft Bug
if not hasattr(accelerate.utils.memory, "clear_device_cache"):
    accelerate.utils.memory.clear_device_cache = lambda: None

# Fix für fehlende interne Transformer-Variablen in rfdetr (v4.48 kompatibel)
if not hasattr(transformers.utils, "torch_int"):
    transformers.utils.torch_int = (torch.int8, torch.int16, torch.int32, torch.int64)
if not hasattr(transformers.utils, "torch_float"):
    transformers.utils.torch_float = (torch.float16, torch.float32, torch.float64, torch.bfloat16)

# Verhindert Fehlermeldungen in neueren Transformer-Versionen
import transformers.utils.import_utils
if not hasattr(transformers.utils, "is_torch_available"):
    transformers.utils.is_torch_available = lambda: True

# -------------------------------------------------------

# Wichtig: Pfade setzen, damit die verifizierten Module gefunden werden
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'ThermalTrack'))
from models.detector import ThermalDetector
from src.tracker import Tracker
from models.pose_engine import ThermalPoseEngine
from utils.benchmark import PipelineBenchmark
# ---------------------------------------------------------------------------
# 1. Konfiguration & Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="ThermalMPT - Online Tracker & Pose Estimation")
    parser.add_argument("--source", type=str, required=True, help="Pfad zum Video (.mp4) oder Bilderordner")
    parser.add_argument("--det-weights", type=str, default="weights/rf-detr_thermal_ema.pth", help="RF-DETR Gewichte")
    parser.add_argument("--pose-weights", type=str, default="weights/vitpose_thermal.pth", help="ViTPose Gewichte")
    parser.add_argument("--device", type=str, default="cuda", help="cuda oder cpu")
    parser.add_argument("--no-show", action="store_true", help="Deaktiviert die Live-Visualisierung")
    parser.add_argument("--save-video", type=str, default="", help="Pfad zum Speichern des Ausgabe-Videos (.avi)")
    parser.add_argument("--export-json", type=str, default="results.json", help="Pfad für den COCO-Style JSON Export")
    parser.add_argument("--use-gstreamer", action="store_true", help="Aktiviert NVDEC Hardware-Decoding auf dem Jetson")
    return parser.parse_args()

def get_gstreamer_pipeline(video_path: str) -> str:
    """Erstellt die GStreamer-Pipeline für den Jetson Hardware-Decoder."""
    return (
        f"filesrc location={video_path} ! qtdemux ! h264parse ! "
        f"nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )

# ---------------------------------------------------------------------------
# 2. Main Online Tracking Loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    logger.info("Starte ThermalMPT Pipeline...")

   # --- Modelle laden ---
    logger.info("Lade Detektor (RF-DETR)...")
    #Alt
    #detector = RFDETRBase(pretrain_weights=args.det_weights)
    #Neu
    detector = ThermalDetector(weights_path=args.det_weights,conf_threshold=0.5)
    # Hinweis: Wir rufen hier kein .to("cpu") oder .eval() auf. 
    # Der RFDETRBase Wrapper regelt das Device und den Status komplett intern!

    logger.info("Initialisiere Tracker...")
    tracker_config = {
        'inactive_patience': 30, 'use_nsa': True, 'nsa_use_square': True,
        'nsa_scale_factor': 0.15, 'use_cw': False, 'cw_score_thresh': 0.4,
        'cw_scale_factor': 2.5, 'init_min_score': 0.6, 'n_dets_for_activation': 1,
        'use_reid': False, 'ema_alpha': 0.9, 'matching_stages': [1],
        'matching_stage_1': {'track_types': ['active', 'tentative'], 'min_score': 0.5, 'metrics': ['iou'], 'weights': [1], 'dist_thresh': 0.8},
        'output_dir': './out', 'detections_dir': './det'
    }
    tracker = Tracker(tracker_config)

    logger.info("Lade Pose Engine (ViTPose)...")
    pose_engine = ThermalPoseEngine(model_path=args.pose_weights, device="cpu")
    # --- Video/Input Setup ---
    if args.use_gstreamer and args.source.endswith(".mp4"):
        pipeline = get_gstreamer_pipeline(args.source)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        logger.info("Nutze GStreamer für NVDEC Hardware-Beschleunigung.")
    else:
        cap = cv2.VideoCapture(args.source)
        logger.info("Nutze Standard OpenCV VideoCapture.")

    if not cap.isOpened():
        logger.error(f"Konnte Videoquelle nicht öffnen: {args.source}")
        sys.exit(1)

    # --- Video Writer Setup ---
    video_writer = None
    if args.save_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))

    # --- JSON Export Setup ---
    coco_results = {"images": [], "annotations": []}
    frame_id = 0
    annotation_id = 0

    logger.info("🚀 Starte Frame-by-Frame Verarbeitung (Online Tracker)")
    
    # NEU: Hier initialisieren wir die Benchmark-Klasse VOR der Schleife
    benchmark = PipelineBenchmark(history_size=30)

    # --- Die Online-Schleife ---
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Ende des Videos erreicht.")
            break
            
        frame_id += 1
        start_time = time.time()
    # Stoppuhr für den gesamten Frame starten
        benchmark.start("total")

        # ---------------------------------------------------
        # 1. Detektion
        # ---------------------------------------------------
        benchmark.start("detektion")
        
        # WICHTIG: Die alte "sv_dets = detector.predict(frame)" Zeile ist jetzt 
        # gelöscht, da unser neuer Wrapper das alles intern macht!
        dets_array = detector.detect(frame)
        
        benchmark.stop("detektion")

        # ---------------------------------------------------
        # 2. Tracking Update
        # ---------------------------------------------------
        benchmark.start("tracking")
        
        tracker.step(dets_array)
        tracker_results = tracker.get_results()

        active_tracks = {}
        for trk_id, history in tracker_results.items():
            if frame_id in history:
                active_tracks[trk_id] = history[frame_id][:4].astype(int)
                
        benchmark.stop("tracking")

        # ---------------------------------------------------
        # 3. Pose Estimation
        # ---------------------------------------------------
        benchmark.start("pose")
        
        poses = pose_engine.estimate_poses(frame, active_tracks)
        
        benchmark.stop("pose")

        # ---------------------------------------------------
        # 4. JSON Daten aufbereiten & Visualisieren
        # ---------------------------------------------------
        coco_results["images"].append({"id": frame_id, "file_name": f"frame_{frame_id}.jpg"})
        
        for trk_id, bbox in active_tracks.items():
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            keypoints = poses.get(trk_id, [])
            
            flat_kps = []
            for kp in keypoints:
                flat_kps.extend([kp[0], kp[1], 2 if kp[2] > 0.3 else 1])
            
            coco_results["annotations"].append({
                "id": annotation_id,
                "image_id": frame_id,
                "track_id": trk_id,
                "category_id": 1,
                "bbox": [int(x1), int(y1), int(w), int(h)],
                "keypoints": flat_kps
            })
            annotation_id += 1

            # ---------------------------------------------------
            # Visualisierung ins Bild zeichnen (OHNE IDs, MIT SKELETT)
            # ---------------------------------------------------
            if not args.no_show or args.save_video:
                # 1. Bounding Box zeichnen (Grün, dünn)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                # Wir löschen die cv2.putText Zeile für die ID komplett!
                
                # 2. Knochen (Linien) zeichnen
                for p1, p2 in COCO_SKELETON:
                    # Prüfen, ob wir genug Keypoints haben (Sicherheitscheck)
                    if p1 < len(keypoints) and p2 < len(keypoints):
                        x1_k, y1_k, conf1 = keypoints[p1]
                        x2_k, y2_k, conf2 = keypoints[p2]
                        
                        # Nur zeichnen, wenn BEIDE Gelenke sicher erkannt wurden
                        if conf1 > 0.3 and conf2 > 0.3:
                            cv2.line(frame, (int(x1_k), int(y1_k)), (int(x2_k), int(y2_k)), (255, 255, 0), 2) # Cyan Linien
                
                # 3. Gelenke (Punkte) ÜBER die Linien zeichnen
                for (kx, ky, conf) in keypoints:
                    if conf > 0.3:
                        cv2.circle(frame, (int(kx), int(ky)), 4, (0, 0, 255), -1) # Rote Punkte
    # Stoppuhr für den gesamten Frame stoppen
        benchmark.stop("total")
        
        # NEU: Loggt die Statistik-Tabelle alle 30 Frames in die Konsole
        benchmark.log_stats(frame_id)              

        # FPS Berechnung
        process_time = time.time() - start_time
        fps_current = 1.0 / process_time if process_time > 0 else 0
        
        if frame_id % 10 == 0:
            logger.info(f"Frame {frame_id} | Tracks: {len(active_tracks)} | FPS: {fps_current:.2f}")

        # Frame speichern / anzeigen
        if args.save_video:
            video_writer.write(frame)
            
        if not args.no_show:
            cv2.putText(frame, f"FPS: {fps_current:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("ThermalMPT - Live", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Abbruch durch Benutzer (q gedrückt).")
                break

    # --- Aufräumen & Export ---
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    logger.info(f"Speichere COCO-Style JSON nach {args.export_json}...")
    with open(args.export_json, 'w') as f:
        json.dump(coco_results, f, indent=4)
        
    logger.info("✅ Pipeline erfolgreich beendet!")

if __name__ == "__main__":
    import torch # Import hier für den Check in Zeile 58
    main()