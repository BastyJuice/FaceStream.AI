import re
import os
import face_recognition
import numpy as np


class FaceLoader:
    def __init__(self, config_manager=None):
        img_dir = os.path.join('/data', 'knownfaces')
        self.config_manager = config_manager
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces(img_dir)

    @staticmethod
    def _normalize_person_name(name: str) -> str:
        # Strip optional quotes and whitespace
        name = (name or "").strip().strip('"').strip("'").strip()
        return name

    @staticmethod
    def _name_from_filename(filename: str) -> str:
        # Backwards compatible: allow filenames like "name.v1.jpg" but return "name"
        stem = os.path.splitext(os.path.basename(filename))[0]
        stem = FaceLoader._normalize_person_name(stem)
        stem = re.sub(r"\.v\d+$", "", stem, flags=re.IGNORECASE)
        return stem

    def load_known_faces(self, directory: str):
        try:
            if not os.path.isdir(directory):
                return

            for entry in sorted(os.listdir(directory)):
                entry_path = os.path.join(directory, entry)

                # New mode: subfolder per person
                if os.path.isdir(entry_path):
                    person_name = self._normalize_person_name(entry)
                    if not person_name:
                        continue
                    for filename in sorted(os.listdir(entry_path)):
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                            file_path = os.path.join(entry_path, filename)
                            self._load_single_face(file_path, person_name)
                    continue

                # Legacy mode: flat files in /knownfaces
                if entry.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = entry_path
                    person_name = self._name_from_filename(entry)
                    self._load_single_face(file_path, person_name)

        except FileNotFoundError:
            print(f"Directory {directory} not found.")
        except Exception as e:
            print(f"Failed to load known faces: {e}")

    def _load_single_face(self, file_path: str, person_name: str):
        try:
            image = face_recognition.load_image_file(file_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(person_name)
        except Exception as e:
            print(f"Failed to load face '{file_path}': {e}")

    def get_name(self, face_encoding):
        if not self.known_face_encodings:
            return "Unknown"
        distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            threshold = 0.55
            if self.config_manager is not None:
                try:
                    threshold = float(self.config_manager.get('face_match_threshold', threshold))
                except Exception:
                    pass
            if distances[best_match_index] < threshold:
                return self.known_face_names[best_match_index]
        return "Unknown"
