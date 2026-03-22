import cv2
import threading
import logging
import time
import queue
import os
import json


class CameraManager(threading.Thread):
    def __init__(
        self,
        frame_queue,
        camera_url,
        output_size=(640, 480),
        max_retries=15,
        config_manager=None,
    ):
        super().__init__(daemon=True)
        self.camera_url = camera_url
        self.output_size = output_size
        self.frame_queue = frame_queue
        self.capture = None
        self.running = True
        self.max_retries = max_retries
        self.config_manager = config_manager
        self.trigger_file = os.path.join("/data", "manual_trigger.json")

        # Trigger-Übergang erkennen (OFF -> ON)
        self._last_trigger_active = False

        # Reconnect-/Timing-Steuerung
        self._reconnect_delay = 2.0
        self._read_fail_delay = 0.2
        self._idle_sleep = 0.01
        self._initial_flush_frames = 10

        # Letztes Live-Frame zwischenspeichern
        self._latest_frame = None
        self._latest_frame_ts = 0.0

        # Ausgabe begrenzen, damit nicht unnötig jedes Kameraframe in die Queue gedrückt wird.
        self._last_push_ts = 0.0
        self._default_active_fps = 5.0

        # Letzter gelesener Trigger-FPS-Wert (nur für Logging/Debug)
        self._last_effective_trigger_fps = None

    def _stream_suspend_enabled(self) -> bool:
        try:
            return bool(
                self.config_manager
                and self.config_manager.get("enable_stream_suspend", False)
            )
        except Exception:
            return False

    def _read_trigger_data(self):
        try:
            if not os.path.exists(self.trigger_file):
                return None

            with open(self.trigger_file, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f) or {}

            now = time.time()
            triggered_at = float(data.get("timestamp", 0.0))
            duration = float(data.get("duration", 0.0))
            duration = max(0.0, min(duration, 120.0))

            grace = 10.0
            if self.config_manager:
                try:
                    grace = float(
                        self.config_manager.get("stream_suspend_grace_seconds", 10) or 0
                    )
                except Exception:
                    grace = 10.0

            grace = max(0.0, min(grace, 600.0))
            active_until = triggered_at + duration + grace
            recognition_until = triggered_at + duration

            try:
                fps = float(data.get("fps", self._default_active_fps))
            except Exception:
                fps = self._default_active_fps
            fps = max(0.1, min(fps, 20.0))

            return {
                "now": now,
                "triggered_at": triggered_at,
                "duration": duration,
                "grace": grace,
                "active_until": active_until,
                "recognition_until": recognition_until,
                "fps": fps,
            }

        except Exception as e:
            logging.debug(f"Trigger read failed: {e}")
            return None

    def _trigger_active(self) -> bool:
        data = self._read_trigger_data()
        if not data:
            return False
        return data["now"] <= data["active_until"]

    def _get_active_fps(self) -> float:
        """
        Bestimmt die effektive Push-Rate für den CameraManager.

        Normalbetrieb:
            - nutzt camera_manager_active_fps (oder Default 5 FPS)

        Manueller Trigger:
            - 1..5 FPS  -> Mindestwert 4 FPS für funktionierendes Best-of
            - >5..20 FPS -> strikt 1:1 übernehmen
        """
        fps = self._default_active_fps

        try:
            if self.config_manager:
                value = self.config_manager.get("camera_manager_active_fps", fps)
                fps = float(value)
        except Exception:
            pass

        fps = max(0.5, min(fps, 30.0))

        trigger_data = self._read_trigger_data()
        if trigger_data and trigger_data["now"] <= trigger_data["active_until"]:
            trigger_fps = float(trigger_data["fps"])

            if 1.0 <= trigger_fps <= 5.0:
                effective_fps = max(trigger_fps, 4.0)
            else:
                effective_fps = trigger_fps

            effective_fps = max(0.5, min(effective_fps, 20.0))

            if self._last_effective_trigger_fps != effective_fps:
                logging.info(
                    f"CameraManager trigger FPS active: requested={trigger_fps:.2f}, effective_push_fps={effective_fps:.2f}"
                )
                self._last_effective_trigger_fps = effective_fps

            return effective_fps

        self._last_effective_trigger_fps = None
        return fps

    def _should_push_now(self) -> bool:
        fps = self._get_active_fps()
        min_interval = 1.0 / fps
        now = time.monotonic()

        if (now - self._last_push_ts) >= min_interval:
            self._last_push_ts = now
            return True
        return False

    def _create_capture(self):
        capture = cv2.VideoCapture(self.camera_url)

        if not capture or not capture.isOpened():
            try:
                capture.release()
            except Exception:
                pass
            return None

        try:
            # Kleiner Buffer hilft oft gegen Latenz
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        return capture

    def open_camera(self):
        if not self.camera_url:
            logging.error("Keine Kamera-URL angegeben.")
            raise ValueError("Keine Kamera-URL angegeben")

        self._close_camera()

        for attempt in range(1, self.max_retries + 1):
            if not self.running:
                return

            capture = self._create_capture()
            if capture is not None:
                self.capture = capture
                logging.info("Camera connected successfully.")

                # Nach Reconnect alte Frames verwerfen
                self._flush_frames(self._initial_flush_frames)
                return

            logging.warning(
                f"Verbindungsversuch {attempt}/{self.max_retries} fehlgeschlagen. Neuer Versuch in {self._reconnect_delay:.0f} Sekunden..."
            )
            time.sleep(self._reconnect_delay)

        logging.error("Kamera konnte nach mehreren Versuchen nicht geöffnet werden.")
        raise ValueError("Kamera konnte nicht geöffnet werden")

    def _close_camera(self):
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
        self.capture = None

    def _ensure_camera(self) -> bool:
        if self.capture is not None and self.capture.isOpened():
            return True

        try:
            self.open_camera()
            return self.capture is not None and self.capture.isOpened()
        except Exception as e:
            logging.error(f"Camera open failed: {e}")
            self._close_camera()
            return False

    def _flush_frames(self, count: int):
        if self.capture is None:
            return

        try:
            for _ in range(max(0, count)):
                if not self.capture.grab():
                    break
        except Exception:
            pass

    def _read_latest_frame(self):
        if self.capture is None or not self.capture.isOpened():
            return None

        try:
            ret, frame = self.capture.read()
        except Exception as e:
            logging.warning(f"Read exception, versuche reconnect... {e}")
            self._close_camera()
            return None

        if not ret or frame is None:
            logging.warning("Kein Frame von Kamera erhalten (read), versuche reconnect...")
            self._close_camera()
            return None

        try:
            frame = cv2.resize(frame, self.output_size)
        except Exception as e:
            logging.warning(f"Resize fehlgeschlagen: {e}")
            return None

        self._latest_frame = frame
        self._latest_frame_ts = time.monotonic()
        return frame

    def _push_latest_frame(self, frame):
        if frame is None:
            return

        # Immer nur das neueste Bild in der Queue behalten
        while self.running:
            try:
                self.frame_queue.put(frame, timeout=0.05)
                return
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    logging.debug("Frame queue voll – ältestes Frame verworfen.")
                except queue.Empty:
                    pass
            except Exception as e:
                logging.warning(f"Queue write failed: {e}")
                return

    def run(self):
        while self.running:
            try:
                # Kamera sicherstellen / reconnect
                if not self._ensure_camera():
                    time.sleep(self._reconnect_delay)
                    continue

                # Immer lesen -> Stream bleibt live
                frame = self._read_latest_frame()
                if frame is None:
                    time.sleep(self._read_fail_delay)
                    continue

                trigger_now = self._trigger_active()
                suspend_enabled = self._stream_suspend_enabled()

                # Stream läuft immer weiter, aber Verarbeitung ist suspendiert.
                # Also: lesen JA, an Pipeline weitergeben NUR wenn erlaubt.
                processing_enabled = (not suspend_enabled) or trigger_now

                # Trigger-Übergang OFF -> ON:
                # Queue komplett leeren und sofort das aktuellste Live-Frame freigeben
                if trigger_now and not self._last_trigger_active:

                    # Queue komplett leeren → keine alten Frames mehr
                    while True:
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                        except Exception:
                            break

                    # sofort aktuelles Live-Frame pushen
                    self._push_latest_frame(frame)

                    self._last_push_ts = time.monotonic()
                    self._last_trigger_active = trigger_now

                    time.sleep(self._idle_sleep)
                    continue

                if processing_enabled and self._should_push_now():
                    self._push_latest_frame(frame)

                self._last_trigger_active = trigger_now
                time.sleep(self._idle_sleep)

            except Exception as e:
                logging.exception(f"Error in CameraManager loop: {e}")
                time.sleep(0.5)

        self._close_camera()

    def stop(self):
        self.running = False
        self._close_camera()