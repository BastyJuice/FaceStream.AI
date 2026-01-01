import logging
import socket
import time
import requests
import os
import cv2
import csv
import json
from urllib.parse import quote


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


class EventLogger:
    def __init__(self, log_file, base_url):
        self.log_file = log_file
        self.routePath = f"{base_url}/event-image"  # Basis-URL für Image-Paths

    def log_event(self, timestamp, name, file_name):
        # Zeit und Datum im lokalen Format formatieren
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))

        # Erstellen der vollständigen URL für das Bild
        full_image_url = f"{self.routePath}/{file_name}"

        # Erstellen des Log-Eintrags als Dictionary
        log_entry = {
            "timestamp": formatted_time,
            "name": name,
            "image_path": full_image_url
        }

        # Log-Eintrag in die JSON-Datei schreiben
        with open(self.log_file, 'a') as file:
            # Anhängen des JSON-Strings am Ende der Datei mit einer Zeilenumbruch-Trennung
            file.write(json.dumps(log_entry) + '\n')

        return log_entry


class NotificationService:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.udp_service_url = config_manager.get('udp_service_url', '')
        self.udp_port = config_manager.get('udp_service_port', 0)
        self.use_udp = config_manager.get('use_udp', False)
        self.use_web = config_manager.get('use_web', False)
        self.web_service_url = config_manager.get('web_service_url', '')
        self.use_loxone_vti = config_manager.get('use_loxone_vti', False)
        self.loxone_ip = config_manager.get('loxone_ip', '').strip()
        self.loxone_user = config_manager.get('loxone_user', '').strip()
        self.loxone_pass = config_manager.get('loxone_pass', '').strip()
        self.loxone_text_input = config_manager.get('loxone_text_input', '').strip()
        self.notification_delay = config_manager.get('notification_delay', 60)
        self.image_path = config_manager.get('image_path')
        self.log_file = config_manager.get('log_file')
        self.last_notification_time = {}

        ensure_directory(self.image_path)

        # Ensure event log directory exists so we can create the file lazily on first write.
        try:
            if self.log_file:
                log_dir = os.path.dirname(self.log_file)
                if log_dir:
                    ensure_directory(log_dir)
        except Exception:
            pass

    def format_custom_message(self, message_template, log_entry):
        # Abrufen der Konfiguration für die benutzerdefinierte Nachricht

        # Formatieren des Zeitstempels
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S')
        formatted_date = time.strftime('%Y-%m-%d')

        # Ersetzen der Platzhalter
        message = message_template.replace('[[name]]', log_entry['name'])
        message = message.replace('[[time]]', formatted_time.split(' ')[1])
        message = message.replace('[[date]]', formatted_date)
        message = message.replace('[[image_url]]', log_entry['image_path'])
        message = message.replace('[[timestamp]]', str(time.time()))

        return message

    def notify(self, name, frame, force: bool = False):
        current_time = time.time()
        if force or name not in self.last_notification_time or (
                current_time - self.last_notification_time[name]) > self.notification_delay:
            self.last_notification_time[name] = current_time
            try:
                filename, full_path = self.save_image(frame, name, current_time)
                deleted = cleanup_event_images(
                    self.image_path,
                    self.config_manager.get('eventimage_cleanup_days', 0),
                    logging
                )
                # Keep event_log.json in sync with automatic image cleanup.
                if deleted:
                    prune_event_log(self.log_file, self.image_path, logging)
                log_entry = self.log_event(current_time, name, filename)
                if self.use_web:
                    self.send_http_notification(log_entry)
                if self.use_udp:
                    self.send_udp_message(log_entry)
                if self.use_loxone_vti:
                    self.send_loxone_notification(log_entry)

                logging.info(
                    f"Notification sent for {name} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}")
            except Exception as e:
                logging.exception(f"Notification pipeline failed for {name}: {e}")

    def send_udp_message(self, log_entry):
        message_template = self.config_manager.get('custom_message_udp')
        custom_message = self.format_custom_message(message_template, log_entry).encode('utf-8')
        if self.use_udp:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.sendto(custom_message, (self.udp_service_url, self.udp_port))
                    logging.debug(f"Sent UDP  message to {self.udp_service_url}:{self.udp_port}: {custom_message}")
            except socket.error as sock_err:
                logging.error(f"Socket error occurred: {sock_err}")
            except Exception as e:
                logging.error(f"Failed to send UDP message: {e}")

    import json

    def send_http_notification(self, log_entry):
        message_template = self.config_manager.get('custom_message_http')
        custom_message = self.format_custom_message(message_template, log_entry)
        if self.use_web:
            full_url = self.web_service_url
            headers = {'Content-Type': 'application/json'}

            try:
                # Versuche, custom_message zu einem Python-Dictionary zu parsen
                if isinstance(custom_message, str):
                    custom_message = json.loads(custom_message)
                elif not isinstance(custom_message, dict):
                    logging.error("Custom message is neither a JSON string nor a dictionary.")
                    return
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse custom_message from JSON: {e}")
                return

            try:
                custom_message = json.dumps(custom_message).encode('utf-8')
                response = requests.post(full_url, data=custom_message, headers=headers)
                if response.status_code == 200:
                    logging.info(f"Notification sent to HTTP endpoint {full_url} successfully.")
                else:
                    logging.error(f"Failed to send HTTP notification: {response.status_code} - {response.text}")
                    logging.error(f"Request details: URL={full_url}, Data={custom_message}, Headers={headers}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to send HTTP request: {e}")
                logging.error(f"Request details: URL={full_url}, Data={custom_message}, Headers={headers}")
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")
                logging.error(f"Request details: URL={full_url}, Data={custom_message}, Headers={headers}")


    def send_loxone_notification(self, log_entry):
        """Send recognized name to a Loxone Miniserver via Virtual Text Input (GET request).

        Target pattern:
        http://user:pass@ip/dev/sps/io/Texteingang/NAME
        """
        if not self.use_loxone_vti:
            return

        ip = (self.loxone_ip or self.config_manager.get('loxone_ip', '')).strip()
        user = (self.loxone_user or self.config_manager.get('loxone_user', '')).strip()
        pw = (self.loxone_pass or self.config_manager.get('loxone_pass', '')).strip()
        text_input = (self.loxone_text_input or self.config_manager.get('loxone_text_input', '')).strip()

        if not ip or not text_input:
            logging.error("Loxone Virtual Text Input enabled, but IP or Texteingang is missing.")
            return

        name = str(log_entry.get('name', 'Unknown'))
        # Encode path segments safely
        text_input_enc = quote(text_input, safe='')
        name_enc = quote(name, safe='')

        # Encode credentials for URL usage (Loxone supports http://user:pass@host/...)
        user_enc = quote(user, safe='')
        pw_enc = quote(pw, safe='')

        url = f"http://{user_enc}:{pw_enc}@{ip}/dev/sps/io/{text_input_enc}/{name_enc}"

        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code >= 200 and resp.status_code < 300:
                logging.debug(f"Loxone notification sent: {url}")
            else:
                logging.error(f"Loxone notification failed (HTTP {resp.status_code}): {url}")
        except Exception as e:
            logging.error(f"Failed to send Loxone request: {e} (URL={url})")

    def save_image(self, frame, name, timestamp):
        filename = f"{name}_{int(timestamp)}.jpg"
        filepath = os.path.join(self.image_path, filename)
        ok = cv2.imwrite(filepath, frame)
        if not ok:
            logging.error(f"Failed to write event image: {filepath}")
        return filename, filepath

    def log_event(self, timestamp, name, file_name):
        logger = EventLogger(self.log_file, self.config_manager.get('base_url'))
        return logger.log_event(timestamp, name, file_name)


# --- Event image cleanup ---
def cleanup_event_images(event_image_dir: str, keep_days: int, logger):
    """Delete event image files older than keep_days.

    Returns a list of deleted filenames.
    """
    if not keep_days or keep_days <= 0:
        return []
    cutoff = time.time() - (keep_days * 24 * 60 * 60)
    deleted = []
    try:
        for fn in os.listdir(event_image_dir):
            if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            p = os.path.join(event_image_dir, fn)
            if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                os.remove(p)
                deleted.append(fn)
    except Exception as e:
        logger.warning(f"Event image cleanup failed: {e}")
    return deleted


def prune_event_log(log_file: str, event_image_dir: str, logger=None):
    """Remove event log entries whose image file no longer exists.

    The log format is JSON lines. Each entry contains an 'image_path' URL like
    <base_url>/event-image/<filename>. We keep entries if the referenced file exists.
    Returns the number of removed entries.
    """
    if not log_file:
        return 0
    try:
        if not os.path.exists(log_file):
            return 0
        kept_lines = []
        removed = 0
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                    img_url = str(entry.get('image_path', ''))
                    filename = img_url.rsplit('/', 1)[-1] if '/' in img_url else img_url
                    if filename and os.path.exists(os.path.join(event_image_dir, filename)):
                        kept_lines.append(json.dumps(entry))
                    else:
                        removed += 1
                except Exception:
                    # Drop malformed lines
                    removed += 1

        # Write back atomically
        tmp = log_file + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as out:
            out.write('\n'.join(kept_lines) + ('\n' if kept_lines else ''))
        os.replace(tmp, log_file)
        return removed
    except Exception as e:
        if logger:
            logger.warning(f"Failed to prune event log: {e}")
        return 0