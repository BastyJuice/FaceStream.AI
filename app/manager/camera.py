import cv2
import threading
import logging
import time
import queue
import os
import json


class CameraManager(threading.Thread):
    def __init__(self, frame_queue, camera_url, output_size=(640, 480), max_retries=15, config_manager=None):
        super().__init__()
        self.camera_url = camera_url
        self.output_size = output_size
        self.frame_queue = frame_queue
        self.capture = None
        self.running = True
        self.max_retries = max_retries  # Maximale Anzahl von Verbindungsversuchen
        self.config_manager = config_manager
        self.trigger_file = os.path.join('/data', 'manual_trigger.json')

    def _stream_suspend_enabled(self) -> bool:
        try:
            return bool(self.config_manager and self.config_manager.get('enable_stream_suspend', False))
        except Exception:
            return False

    def _trigger_active(self) -> bool:
        """Return True if manual trigger is active (including grace seconds)."""
        try:
            if not os.path.exists(self.trigger_file):
                return False
            with open(self.trigger_file, 'r') as f:
                data = json.load(f)
            now = time.time()
            triggered_at = float(data.get('timestamp', 0.0))
            duration = float(data.get('duration', 0.0))
            duration = max(0.0, min(duration, 120.0))
            grace = 10.0
            if self.config_manager:
                try:
                    grace = float(self.config_manager.get('stream_suspend_grace_seconds', 10) or 0)
                except Exception:
                    grace = 10.0
            grace = max(0.0, min(grace, 600.0))
            return now <= (triggered_at + duration + grace)
        except Exception as e:
            logging.debug(f"Trigger read failed: {e}")
            return False

    def open_camera(self):
        if not self.camera_url:  # Überprüfe, ob die Kamera-URL leer ist
            logging.error("Keine Kamera-URL angegeben.")
            raise ValueError("Keine Kamera-URL angegeben")
        attempt = 0
        while attempt < self.max_retries and not self.capture:
            self.capture = cv2.VideoCapture(self.camera_url)
            if self.capture.isOpened():
                logging.info("Camera connected successfully.")
                return
            else:
                logging.warning(f"Kann Kamera nicht öffnen, Versuch {attempt + 1}/{self.max_retries}")
                attempt += 1
                try:
                    self.capture.release()
                except Exception:
                    pass
                self.capture = None
                time.sleep(2)  # Wartezeit zwischen den Versuchen
        if not self.capture:
            logging.error("Kamera konnte nach mehreren Versuchen nicht geöffnet werden.")
            raise ValueError("Kamera konnte nicht geöffnet werden")

    def _close_camera(self):
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
        self.capture = None

    def run(self):
        # Don't open camera immediately if suspend mode is enabled
        while self.running:
            try:
                if self._stream_suspend_enabled() and not self._trigger_active():
                    # Suspend: close camera to save CPU/network
                    self._close_camera()
                    time.sleep(0.2)
                    continue

                # Ensure camera is open
                if self.capture is None or not self.capture.isOpened():
                    try:
                        self.open_camera()
                    except Exception as e:
                        logging.error(f"Camera open failed: {e}")
                        self._close_camera()
                        time.sleep(2)
                        continue

                ret, frame = self.capture.read()
                if not ret:
                    logging.warning("Kein Frame von Kamera erhalten, versuche reconnect...")
                    self._close_camera()
                    time.sleep(0.5)
                    continue

                resized_frame = cv2.resize(frame, self.output_size)
                while True:
                    try:
                        self.frame_queue.put(resized_frame, timeout=0.05)
                        break  # Frame erfolgreich hinzugefügt, Schleife verlassen
                    except queue.Full:
                        try:
                            # Versuche, den ältesten Frame zu entfernen, um Platz zu schaffen
                            self.frame_queue.get_nowait()
                            logging.info("Frame queue ist voll. Ältester Frame wurde verworfen.")
                        except queue.Empty:
                            logging.error("Versuch, aus leerer Queue zu lesen obwohl sie voll sein sollte.")

            except Exception as e:
                logging.error(f"Error in CameraManager loop: {e}")
                time.sleep(0.5)

        self._close_camera()

    def stop(self):
        self.running = False
