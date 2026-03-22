import json
import logging
import os
import re
import socket
import time
from typing import Optional
from urllib.parse import quote

import cv2
import requests


def ensure_directory(path):
    if path:
        os.makedirs(path, exist_ok=True)


def _safe_filename_part(value: str) -> str:
    value = str(value or "Unknown").strip()
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._")
    return value or "Unknown"


class EventLogger:
    def __init__(self, log_file, base_url):
        self.log_file = log_file
        base_url = (base_url or "").rstrip("/")
        self.route_path = f"{base_url}/event-image" if base_url else "/event-image"

    def log_event(self, timestamp, name, file_name):
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        full_image_url = f"{self.route_path}/{file_name}"

        log_entry = {
            "timestamp": formatted_time,
            "name": name,
            "image_path": full_image_url
        }

        with open(self.log_file, 'a', encoding='utf-8') as file:
            file.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

        return log_entry


class NotificationService:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.image_path = config_manager.get('image_path')
        self.log_file = config_manager.get('log_file')
        self.last_notification_time = {}

        ensure_directory(self.image_path)

        try:
            if self.log_file:
                log_dir = os.path.dirname(self.log_file)
                if log_dir:
                    ensure_directory(log_dir)
        except Exception:
            pass

    def _cfg(self):
        return {
            'udp_service_url': self.config_manager.get('udp_service_url', ''),
            'udp_port': int(self.config_manager.get('udp_service_port', 0) or 0),
            'use_udp': bool(self.config_manager.get('use_udp', False)),
            'use_web': bool(self.config_manager.get('use_web', False)),
            'web_service_url': self.config_manager.get('web_service_url', ''),
            'use_loxone_vti': bool(self.config_manager.get('use_loxone_vti', False)),
            'loxone_ip': str(self.config_manager.get('loxone_ip', '') or '').strip(),
            'loxone_user': str(self.config_manager.get('loxone_user', '') or '').strip(),
            'loxone_pass': str(self.config_manager.get('loxone_pass', '') or '').strip(),
            'loxone_text_input': str(self.config_manager.get('loxone_text_input', '') or '').strip(),
            'notification_delay': float(self.config_manager.get('notification_delay', 60) or 60),
            'image_path': self.config_manager.get('image_path'),
            'log_file': self.config_manager.get('log_file'),
            'base_url': self.config_manager.get('base_url', ''),
            'custom_message_udp': self.config_manager.get('custom_message_udp', ''),
            'custom_message_http': self.config_manager.get('custom_message_http', ''),
            'eventimage_cleanup_days': int(self.config_manager.get('eventimage_cleanup_days', 0) or 0),
        }

    def cleanup_now(self, keep_days: Optional[int] = None):
        """
        Manuelles Cleanup, z.B. nach Trigger-Ende.
        Löscht Event-Bilder älter als X Tage und pruned danach event_log.json.
        """
        try:
            cfg = self._cfg()
            days = keep_days if keep_days is not None else cfg['eventimage_cleanup_days']

            if not days or days <= 0:
                logging.info("cleanup_now: eventimage_cleanup_days is 0 -> nothing to do.")
                return {"status": "noop", "deleted": 0, "days": days}

            image_path = cfg['image_path']
            log_file = cfg['log_file'] or '/data/event_log.json'

            deleted_files = cleanup_event_images(image_path, days, logging)
            if deleted_files:
                try:
                    prune_event_log(log_file, image_path, logging)
                except Exception as e:
                    logging.warning(f"cleanup_now: prune_event_log failed: {e}")

            logging.info(f"cleanup_now: deleted {len(deleted_files)} file(s) (days={days}).")
            return {"status": "ok", "deleted": len(deleted_files), "days": days}

        except Exception as e:
            logging.warning(f"cleanup_now failed: {e}")
            return {"status": "error", "message": str(e)}

    def format_custom_message(self, message_template, log_entry):
        if not isinstance(message_template, str):
            message_template = ""

        event_ts = time.time()
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event_ts))
        formatted_date = time.strftime('%Y-%m-%d', time.localtime(event_ts))

        message = message_template.replace('[[name]]', str(log_entry.get('name', 'Unknown')))
        message = message.replace('[[time]]', formatted_time.split(' ')[1])
        message = message.replace('[[date]]', formatted_date)
        message = message.replace('[[image_url]]', str(log_entry.get('image_path', '')))
        message = message.replace('[[timestamp]]', str(event_ts))

        return message

    def notify(self, name, frame, force: bool = False):
        cfg = self._cfg()
        current_time = time.time()
        delay = cfg['notification_delay']

        if force or name not in self.last_notification_time or (
            current_time - self.last_notification_time[name]
        ) > delay:
            self.last_notification_time[name] = current_time
            try:
                filename, full_path = self.save_image(frame, name, current_time)
                deleted = cleanup_event_images(
                    cfg['image_path'],
                    cfg['eventimage_cleanup_days'],
                    logging
                )
                if deleted:
                    prune_event_log(cfg['log_file'], cfg['image_path'], logging)

                log_entry = self.log_event(current_time, name, filename)

                if cfg['use_web']:
                    self.send_http_notification(log_entry)

                if cfg['use_udp']:
                    self.send_udp_message(log_entry)

                if cfg['use_loxone_vti']:
                    self.send_loxone_notification(log_entry)

                logging.info(
                    f"Notification sent for {name} at "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}"
                )
            except Exception as e:
                logging.exception(f"Notification pipeline failed for {name}: {e}")

    def send_udp_message(self, log_entry):
        cfg = self._cfg()
        message_template = cfg['custom_message_udp']
        custom_message = self.format_custom_message(message_template, log_entry).encode('utf-8')

        if not cfg['use_udp']:
            return

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.sendto(custom_message, (cfg['udp_service_url'], cfg['udp_port']))
                logging.debug(
                    f"Sent UDP message to {cfg['udp_service_url']}:{cfg['udp_port']}: {custom_message}"
                )
        except socket.error as sock_err:
            logging.error(f"Socket error occurred: {sock_err}")
        except Exception as e:
            logging.error(f"Failed to send UDP message: {e}")

    def send_http_notification(self, log_entry):
        cfg = self._cfg()
        message_template = cfg['custom_message_http']
        custom_message = self.format_custom_message(message_template, log_entry)

        if not cfg['use_web']:
            return

        full_url = cfg['web_service_url']
        headers = {'Content-Type': 'application/json'}

        try:
            if isinstance(custom_message, str):
                custom_message = json.loads(custom_message)
            elif not isinstance(custom_message, dict):
                logging.error("Custom message is neither a JSON string nor a dictionary.")
                return
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse custom_message from JSON: {e}")
            return

        try:
            payload = json.dumps(custom_message, ensure_ascii=False).encode('utf-8')
            response = requests.post(full_url, data=payload, headers=headers, timeout=(3, 10))
            if response.status_code == 200:
                logging.info(f"Notification sent to HTTP endpoint {full_url} successfully.")
            else:
                logging.error(f"Failed to send HTTP notification: {response.status_code} - {response.text}")
                logging.error(f"Request details: URL={full_url}, Headers={headers}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send HTTP request: {e}")
            logging.error(f"Request details: URL={full_url}, Headers={headers}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            logging.error(f"Request details: URL={full_url}, Headers={headers}")

    def send_loxone_notification(self, log_entry):
        """Send recognized name to a Loxone Miniserver via Virtual Text Input."""
        cfg = self._cfg()

        if not cfg['use_loxone_vti']:
            return

        ip = cfg['loxone_ip']
        user = cfg['loxone_user']
        pw = cfg['loxone_pass']
        text_input = cfg['loxone_text_input']

        if not ip or not text_input:
            logging.error("Loxone Virtual Text Input enabled, but IP or Texteingang is missing.")
            return

        name = str(log_entry.get('name', 'Unknown'))
        text_input_enc = quote(text_input, safe='')
        name_enc = quote(name, safe='')

        url = f"http://{ip}/dev/sps/io/{text_input_enc}/{name_enc}"

        try:
            resp = requests.get(url, auth=(user, pw), timeout=5)
            if 200 <= resp.status_code < 300:
                logging.debug(f"Loxone notification sent to {ip} for text input {text_input}.")
            else:
                logging.error(f"Loxone notification failed (HTTP {resp.status_code}) for host {ip}.")
        except Exception as e:
            logging.error(f"Failed to send Loxone request: {e} (host={ip})")

    def send_loxone_name_only(self, name: str) -> None:
        """Send a value to Loxone VTI WITHOUT creating an event log entry."""
        try:
            self.send_loxone_notification({'name': str(name or 'Unknown')})
        except Exception as e:
            logging.warning(f"Loxone VTI send failed: {e}")

    def save_image(self, frame, name, timestamp):
        safe_name = _safe_filename_part(name)
        filename = f"{safe_name}_{int(timestamp)}.jpg"
        filepath = os.path.join(self.image_path, filename)
        ok = cv2.imwrite(filepath, frame)
        if not ok:
            logging.error(f"Failed to write event image: {filepath}")
        return filename, filepath

    def log_event(self, timestamp, name, file_name):
        logger = EventLogger(self.log_file, self.config_manager.get('base_url', ''))
        return logger.log_event(timestamp, name, file_name)


def cleanup_event_images(event_image_dir: str, keep_days: int, logger):
    """Delete event image files older than keep_days."""
    if not keep_days or keep_days <= 0:
        return []
    if not event_image_dir or not os.path.isdir(event_image_dir):
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
    """Remove event log entries whose image file no longer exists."""
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
                        kept_lines.append(json.dumps(entry, ensure_ascii=False))
                    else:
                        removed += 1
                except Exception:
                    removed += 1

        tmp = log_file + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as out:
            out.write('\n'.join(kept_lines) + ('\n' if kept_lines else ''))
        os.replace(tmp, log_file)

        return removed

    except Exception as e:
        if logger:
            logger.warning(f"Failed to prune event log: {e}")
        return 0