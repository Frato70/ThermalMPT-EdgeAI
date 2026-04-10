# AI-Winterschool-PoseEstimation DE
Introduction 
ThermalMPT: Thermal Multi-Person Tracking & Pose Estimation on Edge AI
ThermalMPT ist eine hochoptimierte KI-Pipeline für die Echtzeit-Personenverfolgung und Skeletterkennung in thermischen Infrarot-Videostreams. Entwickelt im Rahmen der AI Winterschool an der Hochschule Karlsruhe (HKA), liegt der Fokus des Projekts auf der Überwindung des "Computing Gap" – der Ausführung rechenintensiver Transformer-Modelle auf ressourcenbeschränkter Edge-Hardware (NVIDIA Jetson Orin Nano).
# Kernmerkmale
Hybride Architektur: Asymmetrisches Routing der Workload – Detektion auf der GPU, Pose Estimation und Tracking auf der CPU zur Vermeidung von VRAM-Engpässen.
State-of-the-Art Modelle: Einsatz von RF-DETR (Transformer-Detektor) und ViTPose (Pose Estimation).
Hardware-Beschleunigung: Implementierung von FP16 Automatic Mixed Precision (AMP) zur Nutzung der NVIDIA Tensor Cores.
Effizientes Tracking: Vektorisierter Kalman-Filter für Online-Tracking mit einer Latenz von < 4ms.
Robustes Memory Management: Einsatz eines 16 GB Swap-Files zur Abfangung von RAM-Spitzen auf dem Jetson-System.
Standardisierter Export: Vollständige Integration des COCO-JSON Standards für Bounding-Boxes und 17-Punkte-Skelette.

# Benchmarks (Jetson Orin Nano 8GB)
Detektion (RF-DETR FP16): ~105 ms
Tracking (Kalman): ~1.5 – 4 ms
Pose Estimation (ViTPose): ~600 ms (pro Person)

# AI-Winterschool-PoseEstimation Eng
Overview
ThermalMPT is a highly optimized AI pipeline for real-time human tracking and skeleton detection in thermal infrared video streams. Developed during the AI Winterschool at Hochschule Karlsruhe (HKA), the project focuses on bridging the "Computing Gap" – running computationally expensive transformer models on resource-constrained edge hardware (NVIDIA Jetson Orin Nano).

# Key Features
Hybrid Architecture: Asymmetric workload routing – Detection on GPU, Pose Estimation and Tracking on CPU to prevent VRAM bottlenecks.
State-of-the-Art Models: Integration of RF-DETR (Transformer Detector) and ViTPose (Pose Estimation).
Hardware Acceleration: Implementation of FP16 Automatic Mixed Precision (AMP) to leverage NVIDIA Tensor Cores.
Efficient Tracking: Fully vectorized Kalman Filter for online tracking with < 4ms latency.
Robust Memory Management: Utilization of a 16 GB swap-file to handle RAM peaks on the Jetson system.

Standardized Export: Full support for COCO-JSON standards for bounding boxes and 17-point skeletons (51-value arrays).
Performance Metrics (Jetson Orin Nano 8GB)
Detection (RF-DETR FP16): ~105 ms
Tracking (Kalman Filter): ~1.5 – 4 ms
Pose Estimation (ViTPose): ~600 ms (per person)

🛠 Installation & Usage / Installation & Nutzung
Requirements / Voraussetzungen
NVIDIA Jetson Orin Nano (or compatible Ubuntu/CUDA system)
Python 3.10+
PyTorch & Torchvision (CUDA enabled)
# Setup

# Repository klonen
git clone https://github.com/Frato70/ThermalMPT-EdgeAI.git
cd AI-Winterschool-PoseEstimation
# Abhängigkeiten installieren
pip install -r requirements.txt
# Gewichte in den /weights Ordner platzieren
# Place weights in the /weights folder

# Run Inference / Inferenz starten
python main.py --source "path/to/video.mp4" --device cuda --save-video

🏗 System Architecture / Systemarchitektur
Preprocessing: Resizing input to 512x384 to optimize VRAM throughput.
Detection: RF-DETR identifies persons and generates bounding boxes (GPU).
Tracking: Kalman Filter assigns persistent IDs via IoU matching (CPU).
Pose Estimation: Top-down ViTPose extracts 17 keypoints for each tracked person (CPU).
Output: Global vector reprojection and COCO-JSON standardization.


👥 Author / Autor
Agahan Yuldashev – M.Sc. EU4M Mechatronic Engineering (HKA)
