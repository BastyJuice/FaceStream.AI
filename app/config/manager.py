import json
import os
import time

data_folder = '/data'
known_faces_folder = os.path.join(data_folder, 'knownfaces')
config_file = os.path.join(data_folder, 'config.json')


def initialize_app_structure():
    default_config = {
        'input_stream_url': '',
        'overlay_color': [220, 220, 200],
        'overlay_transparency': 0.5,
        'overlay_border': 1,
        'enable_face_overlay': True,
        'output_width': 640,
        'output_height': 480,
        'notification_delay': 60,  # Zeit in Sekunden
        'custom_message_udp': json.dumps({
            'name': '[[name]]',
            'image_url': '[[image_url]]',
            'time': '[[time]]',
            'date': '[[date]]',
            'timestamp': '[[timestamp]]'
        }),
        'custom_message_http': json.dumps({
            'name': '[[name]]',
            'image_url': '[[image_url]]',
            'time': '[[time]]',
            'date': '[[date]]',
            'timestamp': '[[timestamp]]'
        }),
        'use_udp': False,
        'use_web': False,
        'use_loxone_vti': False,
        'loxone_ip': '',
        'loxone_user': '',
        'loxone_pass': '',
        'loxone_text_input': '',
        'web_service_url': '',
        'udp_service_port': 0,
        'udp_service_url': '',
        # If enabled, Face Recognition runs automatically every N frames.
        # If disabled, Face Recognition runs only via manual /trigger.
        'enable_stream_suspend': False,
        'stream_suspend_grace_seconds': 10,
        'enable_face_recognition_interval': True,
        'face_recognition_interval': 60,
        'face_scale_factor': 0.75,
        # Upsample factor for face detection; helps detect smaller faces.
        # 0 = none, 1–2 = common, 3 = heavy (CPU expensive)
        'face_upsample_times': 1,
        'face_detection_model': 'hog',
        'face_match_threshold': 0.55,
        'enable_clahe': False,
        'enable_blur_filter': False,
        'blur_threshold': 100.0,
        'eventimage_cleanup_days': 0,
        'image_path': os.path.join('/data', 'saved_faces'),
        'log_file': os.path.join('/data', 'event_log.json')
    }
    return default_config


class ConfigManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.config = {}
        self._mtime = None
        # Throttle reload checks to avoid filesystem syscalls in hot paths.
        # Live reload stays functional, but we won't stat() the config file on every get().
        self.reload_interval = 1.0  # seconds
        self._last_check_ts = 0.0
        self.load_config()

    def load_config(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Config file {self.filepath} was not found.")
        try:
            with open(self.filepath, 'r') as json_file:
                self.config = json.load(json_file)
                try:
                    self._mtime = os.path.getmtime(self.filepath)
                except OSError:
                    self._mtime = None
                if 'eventimage_cleanup_days' not in self.config:
                    self.config['eventimage_cleanup_days'] = 0
                # New options (backwards compatible)
                if 'enable_stream_suspend' not in self.config:
                    self.config['enable_stream_suspend'] = False
                if 'stream_suspend_grace_seconds' not in self.config:
                    self.config['stream_suspend_grace_seconds'] = 10
                if 'face_upsample_times' not in self.config:
                    self.config['face_upsample_times'] = 1
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error reading config file {self.filepath}: {e.msg}")

    def save_config(self):
        try:
            with open(self.filepath, 'w') as json_file:
                json.dump(self.config, json_file, indent=4)
        except Exception as e:
            raise IOError(f"Error saving config file '{self.filepath}': {e}")

    def _reload_if_changed(self):
        """Reload config from disk if the file changed (throttled)."""
        now = time.monotonic()
        # Only check mtime every reload_interval seconds
        if (now - self._last_check_ts) < self.reload_interval:
            return
        self._last_check_ts = now

        try:
            mtime = os.path.getmtime(self.filepath)
        except OSError:
            return
        if self._mtime is None or mtime != self._mtime:
            self.load_config()

    def get(self, key, default=None):
        self._reload_if_changed()
        return self.config.get(key, default)

    def get_snapshot(self):
        """Return a stable (shallow) copy of the current config after a throttled reload check."""
        self._reload_if_changed()
        # Shallow copy is enough because we mostly store primitives/lists.
        return dict(self.config)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()

    def hex_to_rgb(self, hex_color):
        """Converts a Hex color value to an RGB tuple."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:  # Handles shorthand like #FFF
            hex_color = ''.join([c * 2 for c in hex_color])
        try:
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            raise ValueError("Invalid hex color format")

    def rgb_to_hex(self, rgb_color):
        """Konvertiert ein RGB-Tupel in einen Hex-Farbwert."""
        return '#{:02x}{:02x}{:02x}'.format(*rgb_color)

    def get_rgba_overlay(self):
        """Calculates the RGBA value for the overlay based on the overlay color in the configuration."""
        try:
            rgb_color = self.get('overlay_color', [220, 220, 200])  # Default color if none specified
            alpha = 1 - self.get('overlay_transparency', 0.5)
            rgba_color = 'rgba({}, {}, {}, {})'.format(*rgb_color, alpha)
            return rgba_color
        except Exception as e:
            raise ValueError("Error calculating RGBA overlay: {}".format(e))