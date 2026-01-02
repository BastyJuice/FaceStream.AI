import threading
import logging
import cv2
import face_recognition
import queue
import time
import os
import json


class FrameProcessor(threading.Thread):
    def __init__(self, frame_queue, processed_frame_queue, face_loader, config_manager, notification_service):
        super().__init__()
        self.frame_queue = frame_queue
        self.processed_frame_queue = processed_frame_queue
        self.config_manager = config_manager
        config_manager.load_config()
        self.overlay_transparency = config_manager.get('overlay_transparency')
        self.overlay_color = config_manager.get('overlay_color')
        self.enable_face_recognition_interval = config_manager.get('enable_face_recognition_interval', True)
        self.face_recognition_interval = config_manager.get('face_recognition_interval')
        self.face_loader = face_loader
        self.running = True
        self.trackers = []
        self.notification_service = notification_service
        self.frame_count = 0  # Zähler für die Frame-Intervalle
        # Manual trigger state
        self.trigger_file = os.path.join('/data', 'manual_trigger.json')
        self._trigger_mtime = 0.0
        self._trigger_active_until = 0.0
        self._trigger_next_allowed = 0.0
        self._trigger_fps = 0.0
        self._trigger_stop_on_match = False
        self._trigger_force_notify_pending = False
        self._trigger_recognition_until = 0.0
        self._trigger_start_unknown_sent = False
        self._trigger_final_event_sent = False
        self._last_trigger_active_flag = False
        self._trigger_saw_face = False

        # Fast config snapshot for the realtime loop (avoid filesystem stat() in hot path).
        self._cfg = {}
        self._cfg_refresh_interval = 1.0  # seconds
        self._cfg_next_refresh = 0.0

        # Prime snapshot once so we don't have to wait for the first interval.
        try:
            self._cfg = self.config_manager.get_snapshot()
        except Exception:
            self._cfg = {}


    def _refresh_cfg_if_needed(self):
        """Refresh cached config snapshot at most every _cfg_refresh_interval seconds."""
        now = time.monotonic()
        if now < self._cfg_next_refresh:
            return
        try:
            cfg = self.config_manager.get_snapshot()
        except Exception:
            cfg = self._cfg or {}
        self._cfg = cfg
        self._cfg_next_refresh = now + self._cfg_refresh_interval

        # Keep frequently used fields in sync (still cheap because this runs ~1x/sec)
        self.overlay_transparency = cfg.get('overlay_transparency', self.overlay_transparency)
        self.overlay_color = cfg.get('overlay_color', self.overlay_color)
        self.enable_face_recognition_interval = cfg.get('enable_face_recognition_interval', self.enable_face_recognition_interval)
        self.face_recognition_interval = cfg.get('face_recognition_interval', self.face_recognition_interval)


    
    def _refresh_trigger(self):
        """Reload manual trigger file if changed and update trigger window state."""
        try:
            if not os.path.exists(self.trigger_file):
                return
            mtime = os.path.getmtime(self.trigger_file)
            if mtime <= self._trigger_mtime:
                return
            self._trigger_mtime = mtime

            with open(self.trigger_file, 'r', encoding='utf-8', errors='replace') as f:
                data = json.load(f) or {}

            now = time.time()
            triggered_at = float(data.get('timestamp', now))
            duration = float(data.get('duration', 5))
            fps = float(data.get('fps', 3))

            som = data.get('stop_on_match', 0)
            if isinstance(som, str):
                stop_on_match = som.strip().lower() in ('1', 'true', 'yes', 'on')
            else:
                stop_on_match = bool(som)

            force_notify = data.get('force_notify', True)

            # Clamp
            duration = max(0.5, min(duration, 120.0))
            fps = max(0.1, min(fps, 10.0))

            # Grace is used by streaming/overlay logic; recognition/fallback uses duration only
            grace = float(self.config_manager.get('stream_suspend_grace_seconds', 10) or 0)

            self._trigger_active_until = triggered_at + duration + max(0.0, grace)
            self._trigger_recognition_until = triggered_at + duration
            self._trigger_fps = fps
            self._trigger_next_allowed = 0.0  # allow immediately
            self._trigger_stop_on_match = stop_on_match
            self._trigger_force_notify_pending = bool(force_notify)

            # Reset per-trigger state
            self._trigger_start_unknown_sent = False
            self._trigger_final_event_sent = False
            self._trigger_saw_face = False

            logging.info(
                f"Manual trigger activated: duration={duration}s fps={fps} stop_on_match={stop_on_match} force_notify={force_notify}"
            )
        except Exception as e:
            logging.error(f"Failed to refresh manual trigger: {e}")

    
    
    def _stop_trigger_systemwide(self):
        """Stop the manual trigger for all components by clearing state and removing the trigger file."""
        self._trigger_active_until = 0.0
        self._trigger_recognition_until = 0.0
        try:
            if os.path.exists(self.trigger_file):
                os.remove(self.trigger_file)
        except Exception as e:
            logging.warning(f"Failed to remove trigger file: {e}")

    def _apply_clahe(self, frame_bgr):
        """Optional contrast enhancement for low-light scenes."""
        try:
            ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y2 = clahe.apply(y)
            merged = cv2.merge((y2, cr, cb))
            return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        except Exception:
            return frame_bgr

    def _blur_score(self, frame_bgr):
        """Higher means sharper."""
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    def _create_tracker(self):
        """Create an OpenCV tracker with fallbacks for environments without opencv-contrib."""
        for ctor in ("TrackerKCF_create", "TrackerCSRT_create", "TrackerMOSSE_create", "TrackerMIL_create"):
            fn = getattr(cv2, ctor, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    continue
        return None

    def run(self):
        # IMPORTANT: Never let this thread die silently. Any exception here kills face recognition,
        # notifications, snapshots and event log updates.
        while self.running:
            try:
                # Use a timeout so stop() can terminate the thread even when the
                # frame queue is empty (otherwise .get() blocks forever).
                frame = self.frame_queue.get(timeout=0.5)
                if frame is not None:
                    self._refresh_trigger()
                    self._refresh_cfg_if_needed()
                    now = time.time()
                    trigger_active = now <= self._trigger_active_until
                    
                    # Trigger-Ende erkennen (ON -> OFF) und Cleanup laufen lassen
                    if (not trigger_active) and self._last_trigger_active_flag:
                        try:
                            if hasattr(self.notification_service, 'cleanup_now'):
                                self.notification_service.cleanup_now()
                        except Exception as e:
                            logging.warning(f"Post-trigger cleanup failed: {e}")

                    # Status für nächsten Loop merken
                    self._last_trigger_active_flag = trigger_active

                    # On manual trigger start: send a *silent* Unknown to Loxone only (no event log),
                    # so Loxone doesn't keep an old name displayed.
                    if trigger_active and (not self._trigger_start_unknown_sent):
                        try:
                            # Loxone-only send (no event log)
                            if hasattr(self.notification_service, 'send_loxone_name_only'):
                                self.notification_service.send_loxone_name_only('Unknown')
                        except Exception:
                            pass
                        self._trigger_start_unknown_sent = True

                    # If recognition duration elapsed and we still haven't sent a final event:
                    # send exactly one Unknown *only if we saw at least one face during the trigger*.
                    if trigger_active and (now >= self._trigger_recognition_until) and (not self._trigger_final_event_sent):
                        if self._trigger_saw_face:
                            try:
                                self.notification_service.notify('Unknown', frame, force=True)
                            except Exception as e:
                                logging.exception(f"Notification failed for Unknown: {e}")
                            self._trigger_final_event_sent = True
                        self._stop_trigger_systemwide()
                        trigger_active = False
                    trigger_allow = trigger_active and (now >= self._trigger_next_allowed)

                    if trigger_allow:
                        # Throttle recognition during trigger window
                        self._trigger_next_allowed = now + (1.0 / self._trigger_fps)
                        processed_frame = self.process_frame(frame, trigger_active=True)
                    elif self.enable_face_recognition_interval and (self.frame_count % self.face_recognition_interval == 0):
                        processed_frame = self.process_frame(frame, trigger_active=False)
                    else:
                        # Always update trackers between detections
                        processed_frame = self.update_trackers(frame)

                    try:
                        self.processed_frame_queue.put_nowait(processed_frame)
                    except queue.Full:
                        # Drop older processed frames if the queue is full
                        while not self.processed_frame_queue.empty():
                            try:
                                self.processed_frame_queue.get_nowait()
                            except queue.Empty:
                                break
                        logging.debug("Processed frame queue was full and has been cleared.")
                        try:
                            self.processed_frame_queue.put_nowait(processed_frame)
                        except Exception:
                            pass

                self.frame_count += 1
                if self.frame_count % 100 == 0:
                    queue_size = self.frame_queue.qsize()
                    logging.debug(f"Current queue size: {queue_size}; processed frames: {self.frame_count}")

                # Avoid overflow
                if self.frame_count >= 1000000:
                    self.frame_count = 0

            except queue.Empty:
                continue
            except Exception as e:
                logging.exception(f"Unhandled exception in FrameProcessor thread: {e}")
                time.sleep(0.2)
                continue

    def stop(self):
        self.running = False

    def process_frame(self, frame, trigger_active: bool = False):
        start_time = time.time()
        try:
            self._refresh_cfg_if_needed()
            cfg = self._cfg or {}

            scale_factor = float(cfg.get('face_scale_factor', 0.75))
            # Clamp to reasonable range
            if scale_factor < 0.25:
                scale_factor = 0.25
            if scale_factor > 1.0:
                scale_factor = 1.0
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

            # Optional blur filter: skip recognition on very blurry frames
            if cfg.get('enable_blur_filter', False):
                try:
                    blur_threshold = float(cfg.get('blur_threshold', 100.0))
                except Exception:
                    blur_threshold = 100.0
                score = self._blur_score(small_frame)
                if score < blur_threshold:
                    # No detection this round; just return tracker-updated frame
                    return frame

            # Optional low-light enhancement
            if cfg.get('enable_clahe', False):
                small_frame = self._apply_clahe(small_frame)

            # Convert small frame to RGB from BGR, which OpenCV uses
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Reset trackers on new detection
            self.trackers = []
            
            # Detect faces
            model = str(cfg.get('face_detection_model', 'hog')).lower().strip()
            if model not in ('hog', 'cnn'):
                model = 'hog'
            # Upsampling helps detect smaller faces (at the cost of CPU).
            # 0 = no upsample, 1–2 = common, 3 = heavy.
            try:
                upsample = int(cfg.get('face_upsample_times', 1))
            except Exception:
                upsample = 1
            upsample = max(0, min(upsample, 3))

            face_locations = face_recognition.face_locations(
                rgb_small_frame,
                number_of_times_to_upsample=upsample,
                model=model
            )
            if trigger_active and face_locations:
                self._trigger_saw_face = True
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Create a copy of the original frame to draw on
        except Exception as e:
            logging.exception(f"Face detection/encoding failed: {e}")
            return frame

        marked_frame = frame.copy()

        # Manual trigger behavior:
        # Only create images/events if at least one face was detected.
        # (No snapshot/event when trigger fires but no person is in frame.)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = self.face_loader.get_name(face_encoding)
            # Initialize a new tracker for each face
            tracker = self._create_tracker()
            # Convert face location from small frame scale to original scale
            # Skalierung zurücksetzen
            # IMPORTANT: scale_factor is typically not a clean divisor (e.g. 0.75).
            # Using int(1/scale_factor) truncates (1/0.75 -> 1) and breaks the rescaling.
            scale_multiplier = 1.0 / float(scale_factor)
            top = int(round(top * scale_multiplier))
            right = int(round(right * scale_multiplier))
            bottom = int(round(bottom * scale_multiplier))
            left = int(round(left * scale_multiplier))
            bbox = (left, top, right - left, bottom - top)
            if tracker is not None:
                tracker.init(frame, bbox)
                self.trackers.append({'tracker': tracker, 'name': name})
            # Draw rectangles and notify
            if cfg.get('enable_face_overlay', True):
                marked_frame = self.draw_rectangle_with_name(marked_frame, top, right, bottom, left, name)
            # Trigger-aware notification: allow one forced notification per manual trigger
            now = time.time()
            trigger_active = now <= self._trigger_active_until
            
            if trigger_active:
                # During a manual trigger we log exactly ONE event:
                # - known name immediately on first match (and optionally stop_on_match)
                # - otherwise Unknown once at trigger end (handled above)
                if (name != 'Unknown') and (not self._trigger_final_event_sent):
                    force = True
                    if self._trigger_force_notify_pending:
                        self._trigger_force_notify_pending = False
                    try:
                        self.notification_service.notify(name, marked_frame, force=force)
                    except Exception as e:
                        logging.exception(f"Notification failed for {name}: {e}")
                    self._trigger_final_event_sent = True
                    if self._trigger_stop_on_match:
                        self._stop_trigger_systemwide()
            else:
                force = False
                try:
                    self.notification_service.notify(name, marked_frame, force=force)
                except Exception as e:
                    logging.exception(f"Notification failed for {name}: {e}")

            processing_time = time.time() - start_time
            logging.debug(f"Frame processed in {processing_time:.2f} seconds")

        return marked_frame

    def update_trackers(self, frame):
        self._refresh_cfg_if_needed()
        cfg = self._cfg or {}
        new_trackers = []
        updated_frame = frame.copy()  # Erstelle eine Kopie für Updates
        for tracked in self.trackers:
            tracker = tracked['tracker']
            name = tracked['name']
            success, box = tracker.update(updated_frame)
            if success:
                left, top, width, height = [int(v) for v in box]
                right, bottom = left + width, top + height
                # Respect overlay toggle for tracker-only updates as well
                if cfg.get('enable_face_overlay', True):
                    try:
                        updated_frame = self.draw_rectangle_with_name(updated_frame, top, right, bottom, left, name)
                    except Exception as e:
                        logging.debug(f"Failed to draw tracker overlay for {name}: {e}")
                new_trackers.append(tracked)
            else:
                logging.debug(f"Tracking failed for {name}, removing tracker.")
        self.trackers = new_trackers
        return updated_frame

    def draw_rectangle_with_name(self, frame, top, right, bottom, left, name):
        """Draw a semi-transparent filled face box + name label.

        Performance note:
        We blend only the region-of-interest (ROI) of the face box instead of the full frame.
        This avoids a full-frame copy + addWeighted() on every face and reduces CPU significantly.
        """
        try:
            h, w = frame.shape[:2]

            # Clamp coordinates to frame bounds
            left_i = max(0, min(int(left), w - 1))
            right_i = max(0, min(int(right), w))
            top_i = max(0, min(int(top), h - 1))
            bottom_i = max(0, min(int(bottom), h))

            if right_i <= left_i or bottom_i <= top_i:
                return frame

            border_color = (255, 255, 255)  # white
            border_thickness = 1
            transparency = float(self.overlay_transparency)
            if transparency < 0.0:
                transparency = 0.0
            if transparency > 1.0:
                transparency = 1.0

            # Convert overlay color from RGB (config) to BGR (OpenCV)
            overlay_color = tuple(int(c) for c in self.overlay_color[::-1])

            # Draw border (outline) directly (cheap)
            cv2.rectangle(
                frame,
                (max(0, left_i - border_thickness), max(0, top_i - border_thickness)),
                (min(w - 1, right_i + border_thickness), min(h - 1, bottom_i + border_thickness)),
                border_color,
                border_thickness,
            )

            # ROI-only alpha blend for filled rectangle
            roi = frame[top_i:bottom_i, left_i:right_i]
            if roi.size == 0:
                return frame

            # Make a solid overlay for the ROI
            overlay = roi.copy()
            overlay[:, :] = overlay_color

            # Keep legacy semantics: transparency=0 => fully colored overlay; transparency=1 => original
            cv2.addWeighted(overlay, 1.0 - transparency, roi, transparency, 0.0, roi)

            # Text
            font_scale = 1.0
            font_thickness = 2

            # Prefer below the box, but if it would go out of bounds, place above.
            text_y = bottom_i + border_thickness + 25
            if text_y > h - 5:
                text_y = max(15, top_i - 10)

            cv2.putText(
                frame,
                str(name),
                (left_i, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
            )
        except Exception as e:
            # Never raise from overlay rendering; return the original frame unchanged.
            logging.debug(f"Failed to draw rectangle with name (ROI overlay): {e}")
            return frame

        return frame
