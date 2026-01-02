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
        self.max_retries = max_retries
        self.config_manager = config_manager
        self.trigger_file = os.path.join('/data', 'manual_trigger.json')

        # Trigger-Übergang erkennen (OFF -> ON)
        self._last_trigger_active = False

    def _stream_suspend_enabled(self) -> bool:
        try:
            return bool(self.config_manager and self.config_manager.get('enable_stream_suspend', False))
        except Exception:
            return False

    def _trigger_active(self) -> bool:
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
        if not self.camera_url:
            logging.error("Keine Kamera-URL angegeben.")
            raise ValueError("Keine Kamera-URL angegeben")

        attempt = 0
        while attempt < self.max_retries and not self.capture:
            self.capture = cv2.VideoCapture(self.camera_url)
            if self.capture.isOpened():
                try:
                    # Buffer klein halten → weniger Latenz (wirkt nicht bei allen Backends, schadet aber nicht)
                    self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                logging.info("Camera connected successfully.")
                return
            else:
                attempt += 1
                logging.warning(f"Verbindungsversuch {attempt} fehlgeschlagen. Neuer Versuch in 2 Sekunden...")
                try:
                    self.capture.release()
                except Exception:
                    pass
                self.capture = None
                time.sleep(2)

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
        while self.running:
            try:
                # Trigger pro Loop nur einmal lesen (Datei-IO reduziert)
                trigger_now = self._trigger_active()
                suspend_enabled = self._stream_suspend_enabled()

                # Suspend-Mode: Verbindung warm halten, ABER Backlog verhindern (Drain)
                if suspend_enabled and not trigger_now:
                    try:
                        if self.capture is None or not self.capture.isOpened():
                            self.open_camera()
                    except Exception as e:
                        logging.error(f"Camera open failed (suspend warmup): {e}")
                        self._close_camera()
                        time.sleep(0.5)
                        self._last_trigger_active = trigger_now
                        continue

                    try:
                        ok = self.capture.grab()  # minimal overhead, no decode
                    except Exception:
                        ok = False

                    if not ok:
                        logging.warning("Grab failed during suspend; reconnecting...")
                        self._close_camera()
                        time.sleep(0.5)
                    else:
                        # WICHTIG: Drain statt 1 FPS keep-alive (sonst Backlog bei 20 FPS Quelle)
                        time.sleep(0.05)  # 20 Hz; alternativ 0.1 für weniger Load

                    self._last_trigger_active = trigger_now
                    continue

                # Kamera sicherstellen
                if self.capture is None or not self.capture.isOpened():
                    try:
                        self.open_camera()
                        # Nach Reconnect alte Frames verwerfen
                        try:
                            for _ in range(10):
                                self.capture.grab()
                        except Exception:
                            pass
                    except Exception as e:
                        logging.error(f"Camera open failed: {e}")
                        self._close_camera()
                        time.sleep(2)
                        self._last_trigger_active = trigger_now
                        continue

                # Trigger-Übergang (OFF -> ON): einmalig kurz flushen, um sofort live zu sein
                if trigger_now and not self._last_trigger_active:
                    t_end = time.monotonic() + 0.4  # 0.3–0.6s bewährt
                    try:
                        while time.monotonic() < t_end:
                            if not self.capture.grab():
                                break
                    except Exception:
                        pass

                # IMMER NEUESTES FRAME HOLEN (Frame-Drop aktiv)
                try:
                    for _ in range(2):  # 2–3 ideal bei MJPEG @ ~20 FPS
                        self.capture.grab()
                except Exception:
                    logging.warning("Grab fehlgeschlagen, versuche reconnect...")
                    self._close_camera()
                    time.sleep(0.5)
                    self._last_trigger_active = trigger_now
                    continue

                ret, frame = self.capture.retrieve()
                if not ret or frame is None:
                    logging.warning("Kein Frame von Kamera erhalten (retrieve), versuche reconnect...")
                    self._close_camera()
                    time.sleep(0.5)
                    self._last_trigger_active = trigger_now
                    continue

                resized_frame = cv2.resize(frame, self.output_size)

                # Queue: immer nur neuestes Frame behalten
                while True:
                    try:
                        self.frame_queue.put(resized_frame, timeout=0.05)
                        break
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                            logging.debug("Frame queue voll – ältestes Frame verworfen.")
                        except queue.Empty:
                            pass

                self._last_trigger_active = trigger_now

            except Exception as e:
                logging.error(f"Error in CameraManager loop: {e}")
                time.sleep(0.5)

        self._close_camera()

    def stop(self):
        self.running = False
