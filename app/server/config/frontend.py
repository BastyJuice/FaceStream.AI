import os
import shutil
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import re
from urllib.parse import urlparse, quote
import logging
import json
import time
import requests
from flask import send_file

# Keep event log consistent when images are deleted via the GUI cleanup action.
from notification.service import prune_event_log

UPLOAD_FOLDER = '/data/knownfaces'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def safe_path(base_dir: str, relative_path: str) -> str:
    """Resolve a user-provided relative path safely under base_dir."""
    base = Path(base_dir).resolve()
    rel = (relative_path or "").lstrip("/").replace("\\", "/")
    candidate = (base / rel).resolve()
    # Ensure candidate is within base
    if base == candidate or str(candidate).startswith(str(base) + os.sep):
        return str(candidate)
    raise ValueError("Invalid path")

def sanitize_person_name(name: str) -> str:
    """Turn user input into a safe folder name while keeping it readable."""
    if name is None:
        return ""
    name = name.strip().strip('"').strip("'").strip()
    # Replace path separators and other problematic chars
    name = re.sub(r"[\\/\x00-\x1f:<>\|\?\*]+", "_", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


def get_known_faces_structure(base_dir: str):
    """Return dict: {person_name: [relative_paths...]}, plus root images under key '__root__'."""
    persons = {}
    root_images = []
    if not os.path.isdir(base_dir):
        return persons, root_images

    for entry in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, entry)
        if os.path.isdir(p):
            person = entry
            imgs = []
            for fn in sorted(os.listdir(p)):
                if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                    imgs.append(f"{person}/{fn}")
            if imgs:
                persons[person] = imgs
            else:
                # still show empty persons in UI
                persons.setdefault(person, [])
        else:
            if entry.lower().endswith(('.jpg', '.jpeg', '.png')):
                root_images.append(entry)

    return persons, root_images



def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_int(value, default, min_value=None, max_value=None):
    try:
        value = int(value)
        if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
            return default
        return value
    except (ValueError, TypeError):
        return default


def validate_bool(value, default):
    logging.debug(value)
    if str(value).lower() in ['true', '1', 't', 'y', 'yes', 'on']:
        return True
    elif str(value).lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        return default


# Funktion zur Validierung von Hex-Farben
def validate_hex_color(value, default):
    if value and re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', value):
        return value
    else:
        return default


# Funktion zur Validierung von URLs
def validate_url(value, default):
    try:
        result = urlparse(value)
        if all([result.scheme, result.netloc]):
            return value
    except:
        pass
    return default


def validate_port(value, default=''):
    try:
        port = int(value)
        if 1 <= port <= 65535:
            return port
        else:
            raise ValueError("Port number out of range")
    except (ValueError, TypeError):
        logging.error(f"Invalid port number provided: {value}, reverting to default: {default}")
        return default


def validate_float(value, default, min_value=0.0, max_value=1.0):
    try:
        value = float(value)
        if value < min_value or value > max_value:
            return default
        return value
    except (TypeError, ValueError):
        return default


class ConfigFrontend:
    def __init__(self, config_manager):
        self.app = Flask(__name__)
        self.config_manager = config_manager
        self.define_routes()

    def define_routes(self):

        @self.app.route('/api/setBaseUrl', methods=['POST'])
        def set_base_url():
            data = request.get_json()
            base_url = data['baseUrl']
            # Setzen der Basis-URL im Konfigurationsmanager
            self.config_manager.set('base_url', base_url)
            self.config_manager.save_config()
            return jsonify({'status': 'URL set successfully', 'baseUrl': base_url})

        @self.app.route('/test_path')
        def test_path():
            try:
                files_list = os.listdir(UPLOAD_FOLDER)
                return jsonify({'files': files_list,
                                'uploadfolder': UPLOAD_FOLDER
                                }), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        
        @self.app.route('/trigger', methods=['GET', 'POST'])
        def manual_trigger():
            # Defaults
            try:
                duration = float(request.args.get('duration', 5))
            except ValueError:
                duration = 5.0
            try:
                fps = float(request.args.get('fps', 3))
            except ValueError:
                fps = 3.0
            stop_on_match = request.args.get('stop_on_match', '0')
            stop_on_match = True if str(stop_on_match) in ('1', 'true', 'True', 'yes', 'on') else False

            # Clamp
            duration = max(0.5, min(duration, 120.0))
            fps = max(0.1, min(fps, 10.0))

            payload = {
                'timestamp': time.time(),
                'duration': duration,
                'fps': fps,
                'stop_on_match': stop_on_match,
                # Force exactly one notification for a known face during this trigger window
                'force_notify': True
            }

            trigger_file = os.path.join('/data', 'manual_trigger.json')
            try:
                with open(trigger_file, 'w') as f:
                    json.dump(payload, f)
                return jsonify({'status': 'ok', 'trigger': payload})
            except Exception as e:
                logging.error(f"Failed to write trigger file {trigger_file}: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500

        

        @self.app.route('/loxone-test', methods=['POST'])
        def loxone_test():
            # Test button: send a fixed name to Loxone Virtual Text Input
            if not self.config_manager.get('use_loxone_vti'):
                return jsonify({'status': 'error', 'message': 'Loxone Virtual Text Input is disabled'}), 400

            ip = (self.config_manager.get('loxone_ip') or '').strip()
            user = (self.config_manager.get('loxone_user') or '').strip()
            pw = (self.config_manager.get('loxone_pass') or '').strip()
            text_input = (self.config_manager.get('loxone_text_input') or '').strip()

            if not ip or not user or not pw or not text_input:
                return jsonify({'status': 'error', 'message': 'Missing Loxone configuration (IP/User/Pass/Texteingang)'}), 400

            name = quote("FaceStream.AI", safe="")
            text_input_enc = quote(text_input, safe="")
            url = f"http://{user}:{pw}@{ip}/dev/sps/io/{text_input_enc}/{name}"

            try:
                r = requests.get(url, timeout=5)
                if r.status_code >= 200 and r.status_code < 300:
                    return jsonify({'status': 'ok', 'url': url}), 200
                return jsonify({'status': 'error', 'message': f'Loxone responded with HTTP {r.status_code}', 'url': url}), 502
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e), 'url': url}), 502
        @self.app.route('/', methods=['GET', 'POST'])
        def index():
            print(self.config_manager.config)
            hex_color = self.config_manager.rgb_to_hex(self.config_manager.get('overlay_color'))
            rgba_overlay = self.config_manager.get_rgba_overlay()
            transparency_value = int(round((self.config_manager.get('overlay_transparency')) * 100))
            persons, root_images = get_known_faces_structure(UPLOAD_FOLDER)
            if request.method == 'POST':
                new_config = {
                    'input_stream_url': validate_url(request.form.get('input_stream_url'), ''),
                    'overlay_color': self.config_manager.hex_to_rgb(request.form.get('overlay_color')),
                    'overlay_transparency': validate_int(request.form.get('overlay_transparency'), 0, 0, 100) / 100,
                    'overlay_border': validate_int(request.form.get('overlay_border'), 1, 1, 4),
                    'enable_face_overlay': validate_bool(request.form.get('enable_face_overlay'), True),
                    'output_width': validate_int(request.form.get('output_width'), 640, 100, 4000),
                    'output_height': validate_int(request.form.get('output_height'), 480, 100, 4000),
                    'custom_message_udp': request.form.get('custom_message_udp',
                                                           '[[name]], spotted at [[time]] on [[date]]!').strip(),
                    'custom_message_http': request.form.get('custom_message_http',
                                                            '[[name]], spotted at [[time]] on [[date]]!').strip(),
                    'use_udp': validate_bool(request.form.get('use_udp'), False),
                    'use_web': validate_bool(
                        request.form.get('use_web'), False),
                    'use_loxone_vti': validate_bool(request.form.get('use_loxone_vti'), False),
                    'loxone_ip': request.form.get('loxone_ip', '').strip(),
                    'loxone_user': request.form.get('loxone_user', '').strip(),
                    'loxone_pass': request.form.get('loxone_pass', '').strip(),
                    'loxone_text_input': request.form.get('loxone_text_input', '').strip(),
                    'web_service_url': request.form.get('web_service_url'),
                    'udp_service_url': request.form.get('udp_service_url'),
                    'udp_service_port': validate_port(request.form.get('udp_service_port')),
                    'notification_delay': validate_int(request.form.get('notification_delay'), 60, 10, max_value=300),
                    'enable_stream_suspend': validate_bool(request.form.get('enable_stream_suspend'), False),
                    'enable_face_recognition_interval': validate_bool(
                        request.form.get('enable_face_recognition_interval'), False
                    ),
                    'face_recognition_interval': validate_int(request.form.get('face_recognition_interval'), 60, 2,
                                                              max_value=300),
                    'face_scale_factor': validate_float(request.form.get('face_scale_factor'), 0.75, 0.25, 1.0),
                    'face_upsample_times': validate_int(request.form.get('face_upsample_times'), 1, 0, max_value=3),
                    'face_detection_model': (request.form.get('face_detection_model') or 'hog').strip().lower(),
                    'face_match_threshold': validate_float(request.form.get('face_match_threshold'), 0.55, 0.30, 0.80),
                    'enable_clahe': validate_bool(request.form.get('enable_clahe'), False),
                    'enable_blur_filter': validate_bool(request.form.get('enable_blur_filter'), False),
                    'blur_threshold': validate_float(request.form.get('blur_threshold'), 100.0, 0.0, 5000.0),
                    'eventimage_cleanup_days': validate_int(request.form.get('eventimage_cleanup_days'), self.config_manager.get('eventimage_cleanup_days', 0), 0, max_value=3650)
                }
                combined = {**self.config_manager.config, **new_config}

                # Normalize tuning options
                if combined.get('face_detection_model') not in ('hog', 'cnn'):
                    combined['face_detection_model'] = 'hog'

                # Mutual exclusivity: stream suspend only allowed when interval is disabled
                if combined.get('enable_face_recognition_interval', False):
                    combined['enable_stream_suspend'] = False

                self.config_manager.config = combined
                self.config_manager.save_config()

                # Neustart des Video-Stream Servers erforderlich, um Änderungen anzuwenden
                with open('/data/signal_file', 'w') as f:
                    f.write("restart")

                return render_template('config_saved.html')
            else:
                return render_template(
                    'config_form.html',
                    config=self.config_manager.config,
                    hex_color=hex_color,
                    transparency_value=transparency_value,
                    rgba_overlay=rgba_overlay,
                    persons=persons,
                    root_images=root_images
                )

        @self.app.route('/upload_faces', methods=['POST'])
        def upload_faces():
            # Dropzone sends the file as 'file'
            if 'file' not in request.files:
                return jsonify({'error': 'No file found in request'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No filename provided'}), 400

            # Person (folder) is required for the new UI; keep legacy behavior if missing
            person_raw = request.form.get('person', '').strip()
            person = sanitize_person_name(person_raw)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)

                if person:
                    person_dir = os.path.join(UPLOAD_FOLDER, person)
                    os.makedirs(person_dir, exist_ok=True)
                    save_path = os.path.join(person_dir, filename)
                else:
                    save_path = os.path.join(UPLOAD_FOLDER, filename)

                file.save(save_path)
                return jsonify({'message': f'File {filename} uploaded successfully'}), 200

            return jsonify({'error': 'Invalid file type'}), 400

        @self.app.route('/create_person', methods=['POST'])
        def create_person():
            data = request.get_json(silent=True) or {}
            person_raw = data.get('person') or request.form.get('person') or ''
            person = sanitize_person_name(person_raw)
            if not person:
                return jsonify({'error': 'Person name is required'}), 400

            person_dir = os.path.join(UPLOAD_FOLDER, person)
            os.makedirs(person_dir, exist_ok=True)
            return jsonify({'message': f'Person {person} angelegt', 'person': person}), 200


        @self.app.route('/list-faces', methods=['GET'])
        def list_faces():
            # Returns the HTML fragment used by the GUI to refresh the persons / thumbnails view.
            persons, root_images = get_known_faces_structure(UPLOAD_FOLDER)
            return render_template('_face_list.html', persons=persons, root_images=root_images)

        @self.app.route('/delete_person/<person>', methods=['POST'])
        def delete_person(person):
            person = sanitize_person_name(person)
            if not person:
                return jsonify({'error': 'Invalid person name'}), 400

            person_dir = os.path.join(UPLOAD_FOLDER, person)
            if not os.path.isdir(person_dir):
                return jsonify({'error': f'Person {person} not found'}), 404

            try:
                shutil.rmtree(person_dir)
                # The GUI uses standard form POSTs for deleting a person.
                # A redirect keeps the UX consistent (full page refresh).
                return jsonify({'status':'ok'})
            except Exception as e:
                return jsonify({'error': f'Error deleting {person}: {str(e)}'}), 500

        @self.app.route('/knownfaces/<path:filename>')
        def knownfaces(filename):
            # Serve known face images (supports subfolders per person)
            try:
                file_path = safe_path(UPLOAD_FOLDER, filename)
            except ValueError:
                return "Invalid path", 400
            if not os.path.isfile(file_path):
                return "Not found", 404
            return send_file(file_path)

        @self.app.route('/delete_image/<path:filename>', methods=['POST'])
        def delete_image(filename):
            # prevent traversal
            try:
                file_path = safe_path(UPLOAD_FOLDER, filename)
            except ValueError:
                return jsonify({'error': 'Invalid path'}), 400
            if os.path.exists(file_path) and os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    return jsonify({'status':'ok'})
                except Exception as e:
                    return jsonify({'error': f'Error deleting {filename}: {str(e)}'}), 500
            return jsonify({'error': f'Image {filename} not found'}), 404

        @self.app.route('/event-image/<filename>')
        def event_image(filename):
            # Basic path traversal protection
            if '..' in filename or filename.startswith('/'):
                return 'Access denied', 403

            base_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(base_dir, self.config_manager.get('image_path'))

            try:
                if not os.path.exists(image_path) or not os.path.isdir(image_path):
                    raise FileNotFoundError('The specified image directory does not exist.')
                return send_from_directory(image_path, filename)
            except FileNotFoundError as e:
                return str(e), 404

        @self.app.route('/last-event-image')
        def last_event_image():
            """Redirect to the most recent event image if available."""
            base_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(base_dir, self.config_manager.get('image_path'))
            if not os.path.exists(image_path) or not os.path.isdir(image_path):
                return 'The specified image directory does not exist.', 404
            imgs = [f for f in os.listdir(image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not imgs:
                return 'No event images available.', 404
            imgs.sort(key=lambda fn: os.path.getmtime(os.path.join(image_path, fn)), reverse=True)
            return redirect(url_for('event_image', filename=imgs[0]))



        @self.app.route('/event_log')
        def event_log():
            """Return event log entries as JSON list for the Event Log tab (Tabulator)."""
            log_file = self.config_manager.get('log_file', '/data/event_log.json')
            entries = []
            try:
                if os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entries.append(json.loads(line))
                            except Exception:
                                # Ignore malformed lines
                                continue
            except Exception as e:
                logging.exception("Failed to read event log file: %s", e)
                return jsonify([])

            # Most recent first
            entries.reverse()
            resp = jsonify(entries)
            # Prevent browser/proxy caching so the Event Log updates reliably without hard refresh.
            resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            resp.headers['Pragma'] = 'no-cache'
            resp.headers['Expires'] = '0'
            return resp
        @self.app.route('/clean_event_images', methods=['POST'])
        def clean_event_images():
            action = (request.form.get('action') or 'clean').strip().lower()

            if action == 'save':
                days_raw = (request.form.get('eventimage_cleanup_days') or '').strip()
                try:
                    days_int = int(days_raw)
                except Exception:
                    days_int = 0
                if days_int < 0:
                    days_int = 0
                if days_int > 3650:
                    days_int = 3650
                self.config_manager.set('eventimage_cleanup_days', days_int)
                self.config_manager.save_config()
                return jsonify({'status': 'ok', 'message': 'Saved.', 'days': days_int})

            # CLEAN uses saved config only (save required)
            days_int = int(self.config_manager.get('eventimage_cleanup_days', 0) or 0)
            if days_int <= 0:
                return jsonify({'status': 'error', 'message': 'Cleanup is disabled (set days > 0 and save first).'}), 400

            base_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(base_dir, self.config_manager.get('image_path'))
            if not os.path.exists(image_path) or not os.path.isdir(image_path):
                return jsonify({'status': 'error', 'message': 'The specified image directory does not exist.'}), 404

            cutoff = time.time() - (days_int * 24 * 60 * 60)
            deleted = 0
            deleted_files = []
            errors = 0
            for fn in os.listdir(image_path):
                if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                p = os.path.join(image_path, fn)
                try:
                    if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                        os.remove(p)
                        deleted += 1
                        deleted_files.append(fn)
                except Exception:
                    errors += 1

            # Prune event_log.json to remove entries whose images were deleted.
            try:
                if deleted_files:
                    log_file = self.config_manager.get('log_file', '/data/event_log.json')
                    prune_event_log(log_file, image_path, logging)
            except Exception:
                logging.exception("Failed to prune event log after cleanup")

            msg = f'Deleted {deleted} image(s).' + (f' ({errors} error(s))' if errors else '')
            return jsonify({'status': 'ok', 'message': msg, 'deleted': deleted, 'errors': errors, 'days': days_int})
    def run(self):
        """Run the configuration frontend (port 5000)."""
        self.app.run(
            host='0.0.0.0',
            port=5000,
            threaded=True,
            use_reloader=False,
            debug=True,
        )