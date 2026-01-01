import threading
import time
import signal
import logging
import os
import json
from flask import Flask, Response, stream_with_context
import cv2
import queue
import numpy as np



def is_manual_trigger_active(config_manager, trigger_file='/data/manual_trigger.json'):
    """Return True if manual trigger is active (including grace seconds)."""
    try:
        if not os.path.exists(trigger_file):
            return False
        with open(trigger_file, 'r') as f:
            data = json.load(f)
        now = time.time()
        triggered_at = float(data.get('timestamp', 0.0))
        duration = float(data.get('duration', 0.0))
        duration = max(0.0, min(duration, 120.0))
        grace = 10.0
        try:
            grace = float(config_manager.get('stream_suspend_grace_seconds', 10) or 0)
        except Exception:
            grace = 10.0
        grace = max(0.0, min(grace, 600.0))
        return now <= (triggered_at + duration + grace)
    except Exception:
        return False


def add_pause_overlay(frame):
    """Draw a straight 'Suspend' overlay with a lightly blurred background."""
    overlay = frame.copy()

    # Light background blur (keep text sharp by drawing after blur)
    try:
        overlay = cv2.GaussianBlur(overlay, (0, 0), 8)
    except Exception:
        # If blur fails for any reason, fall back to unblurred
        overlay = frame.copy()

    h, w = overlay.shape[:2]

    # Semi-transparent dark band behind the text for readability
    band_h = max(80, int(h * 0.18))
    y0 = (h - band_h) // 2
    y1 = y0 + band_h
    band = overlay.copy()
    cv2.rectangle(band, (0, y0), (w, y1), (0, 0, 0), -1)
    overlay = cv2.addWeighted(band, 0.35, overlay, 0.65, 0)

    text_label = "Suspend"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1.2, w / 900.0)
    thickness = max(2, int(w / 500))

    (tw, th), _ = cv2.getTextSize(text_label, font, font_scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2

    # White text with dark outline for contrast (text itself not blurred)
    cv2.putText(overlay, text_label, (x, y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(overlay, text_label, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return overlay
def check_for_restart_signal(signal_file_path, interval=10):
    while True:
        if os.path.exists(signal_file_path):
            logging.info("Signaldatei gefunden. Server wird neu gestartet...")
            os.remove(signal_file_path)
            os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(interval)


class VideoStreamingServer:
    def __init__(self, config_manager, frame_queue):
        self.app = Flask(__name__)
        self.config_manager = config_manager
        self.frame_queue = frame_queue  # Die Warteschlange für Frames
        self.define_routes()
        self.last_request_time = time.time()
        self.active_clients = 0
        self.client_lock = threading.Lock()
        signal_file_path = '/data/signal_file'
        restart_thread = threading.Thread(target=check_for_restart_signal, args=(signal_file_path,))
        restart_thread.daemon = True
        restart_thread.start()

    def define_routes(self):
        @self.app.route('/stream')
        def video_feed():
            with self.client_lock:
                generator = self.start_stream()
                self.active_clients += 1

            def stream():
                try:
                    logging.info("stream gestartet")
                    for frame_chunk in generator:
                        yield frame_chunk
                except Exception as e:
                    logging.error(f"Error during frame generation: {e}")
                finally:
                    with self.client_lock:
                        self.active_clients -= 1
                        if self.active_clients == 0:
                            self.stop_stream()

            return Response(stream_with_context(stream()), mimetype='multipart/x-mixed-replace; boundary=frame')

    def start_stream(self):
        def generate():
            last_frame = None
            while True:
                try:
                    frame = self.frame_queue.get(timeout=1)
                    last_frame = frame

                    _, jpeg = cv2.imencode('.jpg', frame)
                    frame_data = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

                    time.sleep(1.0 / 30)  # target 30 FPS

                except queue.Empty:
                    suspend_enabled = bool(self.config_manager.get('enable_stream_suspend', False))
                    trigger_active = is_manual_trigger_active(self.config_manager)

                    if suspend_enabled and not trigger_active and last_frame is not None:
                        paused = add_pause_overlay(last_frame)
                        _, jpeg = cv2.imencode('.jpg', paused)
                        frame_data = jpeg.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                        time.sleep(0.2)  # ~5 FPS for pause screen
                    else:
                        logging.debug("Warte auf Frames...")
                        time.sleep(0.1)

                except Exception as e:
                    logging.error(f"Error during frame generation: {e}")
                    time.sleep(0.1)

        return generate()


    def stop_stream(self):
        logging.info("stream gestoppt")

    def run(self):
        self.app.run(host='0.0.0.0', port=5001, threaded=True, use_reloader=False)
