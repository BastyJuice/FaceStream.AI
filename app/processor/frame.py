import threading
import logging
import cv2
import face_recognition
import queue
import time
import os
import json
import concurrent.futures
import multiprocessing
import gc
import copy
import numpy as np
from collections import deque


def _clip_box(top, right, bottom, left, width, height):
    top = max(0, min(int(top), height))
    right = max(0, min(int(right), width))
    bottom = max(0, min(int(bottom), height))
    left = max(0, min(int(left), width))
    return top, right, bottom, left


def _roi_blur_score(frame_bgr, face_location):
    try:
        h, w = frame_bgr.shape[:2]
        top, right, bottom, left = _clip_box(*face_location, width=w, height=h)
        roi = frame_bgr[top:bottom, left:right]
        if roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _apply_clahe_to_face_regions(frame_bgr, face_locations):
    try:
        enhanced = frame_bgr.copy()
        h, w = enhanced.shape[:2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        for face_location in face_locations:
            top, right, bottom, left = _clip_box(*face_location, width=w, height=h)
            roi = enhanced[top:bottom, left:right]
            if roi.size == 0:
                continue

            ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y2 = clahe.apply(y)
            merged = cv2.merge((y2, cr, cb))
            enhanced[top:bottom, left:right] = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

        return enhanced
    except Exception:
        return frame_bgr


def _detect_faces_worker(frame, cfg):
    """
    CPU-intensive face detection/encoding in a separate process.
    Returns filtered face_locations, face_encodings and the scale_factor used.
    """
    try:
        scale_factor = float(cfg.get('face_scale_factor', 0.75))

        # Optional trigger-time runtime scale boost. Useful when triggers run at low FPS
        # and we can afford a bit more detail without making the live stream stutter.
        if cfg.get('__trigger_active', False):
            trigger_scale_override = cfg.get('__trigger_scale_override', None)
            if trigger_scale_override is not None:
                try:
                    scale_factor = float(trigger_scale_override)
                except Exception:
                    pass

        if scale_factor < 0.25:
            scale_factor = 0.25
        if scale_factor > 1.0:
            scale_factor = 1.0

        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        working_frame = small_frame

        model = str(cfg.get('face_detection_model', 'hog')).lower().strip()
        if model not in ('hog', 'cnn'):
            model = 'hog'

        try:
            upsample = int(cfg.get('face_upsample_times', 1))
        except Exception:
            upsample = 1
        upsample = max(0, min(upsample, 3))

        rgb_small_frame = working_frame[:, :, ::-1]
        detected_locations = face_recognition.face_locations(
            rgb_small_frame,
            number_of_times_to_upsample=upsample,
            model=model
        )

        if not detected_locations:
            return {
                'face_locations': [],
                'face_encodings': [],
                'scale_factor': scale_factor,
            }

        # Optional ROI blur filter: score only the detected face area, not the full frame.
        filtered_locations = []
        blur_scores = []
        if cfg.get('enable_blur_filter', False):
            try:
                blur_threshold = float(cfg.get('blur_threshold', 100.0))
            except Exception:
                blur_threshold = 100.0

            for loc in detected_locations:
                score = _roi_blur_score(working_frame, loc)
                if score >= blur_threshold:
                    filtered_locations.append(loc)
                    blur_scores.append(score)
        else:
            filtered_locations = list(detected_locations)

        if not filtered_locations:
            return {
                'face_locations': [],
                'face_encodings': [],
                'scale_factor': scale_factor,
            }

        # Optional ROI CLAHE: enhance only detected face regions before encoding.
        encoding_frame = working_frame
        if cfg.get('enable_clahe', False):
            encoding_frame = _apply_clahe_to_face_regions(working_frame, filtered_locations)

        rgb_encoding_frame = encoding_frame[:, :, ::-1]
        face_encodings = face_recognition.face_encodings(rgb_encoding_frame, filtered_locations)

        return {
            'face_locations': filtered_locations,
            'face_encodings': face_encodings,
            'scale_factor': scale_factor,
            'blur_scores': blur_scores,
        }

    except Exception as e:
        return {
            'face_locations': [],
            'face_encodings': [],
            'scale_factor': 1.0,
            'error': str(e),
        }


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
        self.frame_count = 0

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

        # Best-of-frames for manual triggers
        self._trigger_frame_candidates = deque(maxlen=5)
        self._trigger_candidate_poll_interval = 0.05
        self._trigger_min_face_size_for_score = 24  # in scaled candidate frame
        self._trigger_gamma = 1.3

        # Fast config snapshot for the realtime loop
        self._cfg = {}
        self._cfg_refresh_interval = 1.0
        self._cfg_next_refresh = 0.0

        try:
            self._cfg = self.config_manager.get_snapshot()
        except Exception:
            self._cfg = {}

        self._executor_recycle_every = max(1, int(config_manager.get('worker_recycle_every', 200) or 200))
        self._processed_jobs_since_recycle = 0
        self.executor = self._create_executor()

    def _create_executor(self):
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn")
        )

    def _maybe_recycle_executor(self):
        if self._processed_jobs_since_recycle < self._executor_recycle_every:
            return
        try:
            old_executor = self.executor
            self.executor = self._create_executor()
            self._processed_jobs_since_recycle = 0
            old_executor.shutdown(wait=False, cancel_futures=True)
            gc.collect()
            logging.info("Recycled face-detection worker process to cap memory growth.")
        except Exception as e:
            logging.warning(f"Executor recycle failed: {e}")

    def _refresh_cfg_if_needed(self):
        now = time.monotonic()
        if now < self._cfg_next_refresh:
            return
        try:
            cfg = self.config_manager.get_snapshot()
        except Exception:
            cfg = self._cfg or {}
        self._cfg = cfg
        self._cfg_next_refresh = now + self._cfg_refresh_interval

        self.overlay_transparency = cfg.get('overlay_transparency', self.overlay_transparency)
        self.overlay_color = cfg.get('overlay_color', self.overlay_color)
        self.enable_face_recognition_interval = cfg.get(
            'enable_face_recognition_interval',
            self.enable_face_recognition_interval
        )
        self.face_recognition_interval = cfg.get(
            'face_recognition_interval',
            self.face_recognition_interval
        )

    def _refresh_trigger(self):
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

            duration = max(0.5, min(duration, 120.0))
            fps = max(0.1, min(fps, 20.0))

            grace = float(self.config_manager.get('stream_suspend_grace_seconds', 10) or 0)

            self._trigger_active_until = triggered_at + duration + max(0.0, grace)
            self._trigger_recognition_until = triggered_at + duration
            self._trigger_fps = fps
            self._trigger_next_allowed = 0.0
            self._trigger_stop_on_match = stop_on_match
            self._trigger_force_notify_pending = bool(force_notify)

            self._trigger_start_unknown_sent = False
            self._trigger_final_event_sent = False
            self._trigger_saw_face = False
            self._trigger_frame_candidates.clear()

            logging.info(
                f"Manual trigger activated: duration={duration}s fps={fps} "
                f"stop_on_match={stop_on_match} force_notify={force_notify}"
            )
        except Exception as e:
            logging.error(f"Failed to refresh manual trigger: {e}")

    def _stop_trigger_systemwide(self):
        self._trigger_active_until = 0.0
        self._trigger_recognition_until = 0.0
        try:
            if os.path.exists(self.trigger_file):
                os.remove(self.trigger_file)
        except Exception as e:
            logging.warning(f"Failed to remove trigger file: {e}")

    def _apply_clahe(self, frame_bgr):
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
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    def _adjust_gamma(self, image, gamma=1.3):
        try:
            inv_gamma = 1.0 / float(gamma)
            table = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
                dtype="uint8"
            )
            return cv2.LUT(image, table)
        except Exception:
            return image

    def _get_trigger_bestof_plan(self):
        """
        Dynamic best-of tuning for trigger processing.
        Sample timing is handled separately in _collect_trigger_candidates():
        fast on first attempt, slower afterwards for better quality.
        """
        fps = max(0.1, float(self._trigger_fps or 1.0))

        if fps <= 1.0:
            max_candidates = 5
            runtime_scale = 0.90
        elif fps <= 2.0:
            max_candidates = 5
            runtime_scale = 0.85
        elif fps <= 3.0:
            max_candidates = 4
            runtime_scale = 0.80
        else:
            max_candidates = 3
            runtime_scale = None

        return {
            'max_candidates': max_candidates,
            'runtime_scale': runtime_scale,
        }

    def _collect_trigger_candidates(self, first_frame):
        """
        Collect a few fresh frames after trigger allowance.
        Fast on the first attempt, then allow a bit more time for better quality.
        """
        plan = self._get_trigger_bestof_plan()
        max_candidates = int(plan['max_candidates'])

        # schneller Start beim ersten Versuch, danach mehr Qualität
        if not self._trigger_saw_face and not self._trigger_final_event_sent:
            sample_window = 0.12
        else:
            sample_window = 0.35

        candidates = [first_frame]
        deadline = time.monotonic() + sample_window

        while len(candidates) < max_candidates and time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            timeout = min(self._trigger_candidate_poll_interval, remaining)
            if timeout <= 0:
                break
            try:
                next_frame = self.frame_queue.get(timeout=timeout)
            except queue.Empty:
                break
            except Exception:
                break

            if next_frame is not None:
                candidates.append(next_frame)

        return candidates

    def _candidate_face_score(self, frame):
        """
        Score trigger candidates by the quality of the largest detected face,
        not by global image sharpness. This is much more robust for intercom
        scenes with bright background / cars / sky.
        """
        try:
            cfg = self._cfg or {}
            base_scale = float(cfg.get('face_scale_factor', 0.9))
            base_scale = max(0.25, min(base_scale, 1.0))

            # Cheap scoring pass: use a moderate scale and HOG with no upsample.
            plan = self._get_trigger_bestof_plan()
            runtime_scale = plan.get('runtime_scale', None)

            if runtime_scale is not None:
                scale_factor = max(base_scale, float(runtime_scale))
            else:
                scale_factor = base_scale

            small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            rgb = small[:, :, ::-1]

            locations = face_recognition.face_locations(
                rgb,
                number_of_times_to_upsample=0,
                model='hog'
            )

            if not locations:
                return -1.0

            largest = max(
                locations,
                key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3])
            )
            top, right, bottom, left = largest
            face_w = max(1, right - left)
            face_h = max(1, bottom - top)

            if face_w < self._trigger_min_face_size_for_score or face_h < self._trigger_min_face_size_for_score:
                return -0.5

            blur = _roi_blur_score(small, largest)
            face_area = float(face_w * face_h)

            # Brightness score: prefer faces that are not extremely dark.
            roi = small[top:bottom, left:right]
            brightness = 0.0
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                brightness = float(np.mean(gray))

            # Score mix:
            # - blur dominant
            # - face size helpful
            # - brightness small bonus
            return blur + (face_area * 0.02) + (brightness * 0.15)

        except Exception:
            return -1.0

    def _select_best_trigger_frame(self, frame):
        candidates = self._collect_trigger_candidates(frame)
        best_frame = frame
        best_score = -1.0

        self._trigger_frame_candidates.clear()

        for candidate in candidates:
            if candidate is None:
                continue

            score = self._candidate_face_score(candidate)
            self._trigger_frame_candidates.append((score, candidate))

            if score > best_score:
                best_score = score
                best_frame = candidate

        if best_score < 0:
            # Fallback: no face found in the cheap scoring pass.
            # Still prefer the globally sharpest candidate.
            sharpest_frame = frame
            sharpest_score = -1.0
            for candidate in candidates:
                if candidate is None:
                    continue
                score = self._blur_score(candidate)
                if score > sharpest_score:
                    sharpest_score = score
                    sharpest_frame = candidate
            best_frame = sharpest_frame

        return best_frame

    def _build_detection_cfg(self, trigger_active: bool = False):
        cfg = copy.deepcopy(self._cfg or {})
        if not trigger_active:
            return cfg

        plan = self._get_trigger_bestof_plan()
        runtime_scale = plan.get('runtime_scale', None)

        cfg['__trigger_active'] = True
        cfg['__trigger_scale_override'] = runtime_scale

        return cfg

    def _create_tracker(self):
        for ctor in (
            "TrackerKCF_create",
            "TrackerCSRT_create",
            "TrackerMOSSE_create",
            "TrackerMIL_create",
        ):
            fn = getattr(cv2, ctor, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    continue
        return None

    def run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
                if frame is not None:
                    self._refresh_trigger()
                    self._refresh_cfg_if_needed()
                    now = time.time()
                    trigger_active = now <= self._trigger_active_until

                    if (not trigger_active) and self._last_trigger_active_flag:
                        try:
                            if hasattr(self.notification_service, 'cleanup_now'):
                                self.notification_service.cleanup_now()
                        except Exception as e:
                            logging.warning(f"Post-trigger cleanup failed: {e}")

                    self._last_trigger_active_flag = trigger_active

                    if trigger_active and (not self._trigger_start_unknown_sent):
                        try:
                            if hasattr(self.notification_service, 'send_loxone_name_only'):
                                self.notification_service.send_loxone_name_only('Unknown')
                        except Exception:
                            pass
                        self._trigger_start_unknown_sent = True

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
                        self._trigger_next_allowed = now + (1.0 / self._trigger_fps)
                        processed_frame = self.process_frame(frame, trigger_active=True)
                    elif self.enable_face_recognition_interval and (self.frame_count % self.face_recognition_interval == 0):
                        processed_frame = self.process_frame(frame, trigger_active=False)
                    else:
                        processed_frame = self.update_trackers(frame)

                    try:
                        self.processed_frame_queue.put_nowait(processed_frame)
                    except queue.Full:
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
                    logging.debug(
                        f"Current queue size: {queue_size}; processed frames: {self.frame_count}"
                    )

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
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def process_frame(self, frame, trigger_active: bool = False):
        start_time = time.time()
        try:
            self._refresh_cfg_if_needed()

            cfg = self._build_detection_cfg(trigger_active=trigger_active)

            self.trackers = []

            detection_frame = frame
            if trigger_active:
                # erstes Frame sofort ohne Best-of
                if not self._trigger_saw_face and not self._trigger_final_event_sent:
                    detection_frame = frame
                else:
                    detection_frame = self._select_best_trigger_frame(frame)

                detection_frame = self._adjust_gamma(
                    detection_frame,
                    gamma=self._trigger_gamma
                )

            future = self.executor.submit(_detect_faces_worker, detection_frame, cfg)
            try:
                result = future.result(timeout=10)
            finally:
                del future

            self._processed_jobs_since_recycle += 1
            self._maybe_recycle_executor()

            face_locations = result.get('face_locations', [])
            face_encodings = result.get('face_encodings', [])
            scale_factor = float(result.get('scale_factor', 1.0))

            if 'error' in result:
                logging.debug(f"Face detection worker warning: {result['error']}")

            if trigger_active and face_locations:
                self._trigger_saw_face = True

        except Exception as e:
            logging.exception(f"Face detection/encoding failed: {e}")
            return frame

        marked_frame = detection_frame.copy()

        try:
            max_faces = int(cfg.get('max_faces_per_frame', 0))
        except Exception:
            max_faces = 0

        if max_faces > 0:
            face_locations = face_locations[:max_faces]
            face_encodings = face_encodings[:max_faces]

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            try:
                name = self.face_loader.get_name(face_encoding)
            except Exception as e:
                logging.debug(f"Name lookup failed: {e}")
                name = 'Unknown'

            tracker = self._create_tracker()

            scale_multiplier = 1.0 / float(scale_factor)
            top = int(round(top * scale_multiplier))
            right = int(round(right * scale_multiplier))
            bottom = int(round(bottom * scale_multiplier))
            left = int(round(left * scale_multiplier))
            bbox = (left, top, right - left, bottom - top)

            if tracker is not None:
                try:
                    tracker.init(detection_frame, bbox)
                    self.trackers.append({'tracker': tracker, 'name': name})
                except Exception as e:
                    logging.debug(f"Tracker init failed for {name}: {e}")

            if cfg.get('enable_face_overlay', True):
                marked_frame = self.draw_rectangle_with_name(
                    marked_frame,
                    top,
                    right,
                    bottom,
                    left,
                    name
                )

            now = time.time()
            trigger_active_now = now <= self._trigger_active_until

            if trigger_active_now:
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
        updated_frame = frame.copy()

        for tracked in self.trackers:
            tracker = tracked['tracker']
            name = tracked['name']
            success, box = tracker.update(updated_frame)
            if success:
                left, top, width, height = [int(v) for v in box]
                right, bottom = left + width, top + height
                if cfg.get('enable_face_overlay', True):
                    try:
                        updated_frame = self.draw_rectangle_with_name(
                            updated_frame,
                            top,
                            right,
                            bottom,
                            left,
                            name
                        )
                    except Exception as e:
                        logging.debug(f"Failed to draw tracker overlay for {name}: {e}")
                new_trackers.append(tracked)
            else:
                logging.debug(f"Tracking failed for {name}, removing tracker.")

        self.trackers = new_trackers
        return updated_frame

    def draw_rectangle_with_name(self, frame, top, right, bottom, left, name):
        try:
            h, w = frame.shape[:2]

            left_i = max(0, min(int(left), w - 1))
            right_i = max(0, min(int(right), w))
            top_i = max(0, min(int(top), h - 1))
            bottom_i = max(0, min(int(bottom), h))

            if right_i <= left_i or bottom_i <= top_i:
                return frame

            border_color = (255, 255, 255)
            border_thickness = 1
            transparency = float(self.overlay_transparency)
            if transparency < 0.0:
                transparency = 0.0
            if transparency > 1.0:
                transparency = 1.0

            overlay_color = tuple(int(c) for c in self.overlay_color[::-1])

            cv2.rectangle(
                frame,
                (max(0, left_i - border_thickness), max(0, top_i - border_thickness)),
                (min(w - 1, right_i + border_thickness), min(h - 1, bottom_i + border_thickness)),
                border_color,
                border_thickness,
            )

            roi = frame[top_i:bottom_i, left_i:right_i]
            if roi.size == 0:
                return frame

            overlay = roi.copy()
            overlay[:, :] = overlay_color

            cv2.addWeighted(overlay, 1.0 - transparency, roi, transparency, 0.0, roi)

            font_scale = 1.0
            font_thickness = 2

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
            logging.debug(f"Failed to draw rectangle with name (ROI overlay): {e}")
            return frame

        return frame