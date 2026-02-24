import time
import logging
from collections import deque

logger = logging.getLogger(__name__)

class PipelineBenchmark:
    """
    Misst und loggt die Ausführungszeiten der einzelnen Pipeline-Komponenten.
    Nutzt einen gleitenden Durchschnitt (Moving Average) für stabile Werte.
    """
    def __init__(self, history_size=30):
        self.history_size = history_size
        self.times = {
            "detektion": deque(maxlen=history_size),
            "tracking": deque(maxlen=history_size),
            "pose": deque(maxlen=history_size),
            "total": deque(maxlen=history_size)
        }
        self.start_ticks = {}

    def start(self, component: str):
        """Startet den Timer für eine spezifische Komponente."""
        # time.perf_counter() ist viel präziser als time.time()!
        self.start_ticks[component] = time.perf_counter()

    def stop(self, component: str):
        """Stoppt den Timer und speichert die Dauer."""
        if component in self.start_ticks:
            duration = (time.perf_counter() - self.start_ticks[component]) * 1000 # in Millisekunden
            self.times[component].append(duration)

    def get_average(self, component: str) -> float:
        """Gibt den Durchschnitt der letzten N Messungen in ms zurück."""
        if len(self.times[component]) == 0:
            return 0.0
        return sum(self.times[component]) / len(self.times[component])

    def log_stats(self, frame_id: int):
        """Gibt alle 30 Frames eine formatierte Tabelle im Terminal aus."""
        if frame_id % self.history_size == 0:
            det_avg = self.get_average("detektion")
            trk_avg = self.get_average("tracking")
            pose_avg = self.get_average("pose")
            tot_avg = self.get_average("total")
            
            fps = 1000.0 / tot_avg if tot_avg > 0 else 0.0

            logger.info(
                f"\n--- Benchmark (Frame {frame_id}) ---\n"
                f" RF-DETR (Detektion) : {det_avg:.1f} ms\n"
                f" Kalman (Tracking)   : {trk_avg:.1f} ms\n"
                f" ViTPose (Pose)      : {pose_avg:.1f} ms\n"
                f" ---------------------------------\n"
                f" Total pro Frame     : {tot_avg:.1f} ms (~{fps:.1f} FPS)\n"
                f"-----------------------------------"
            )