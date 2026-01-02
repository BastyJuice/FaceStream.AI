import re
import os
import numpy as np


class FaceLoader:
    def __init__(self, config_manager=None):
        img_dir = os.path.join('/data', 'knownfaces')
        self.config_manager = config_manager
        # Store encodings compactly as float32 Nx128 for lower RAM and faster distance calcs.
        self.known_face_encodings = []  # temporary list, finalized to np.ndarray
        self.known_face_names = []
        self.load_known_faces(img_dir)

        # Finalize storage format
        if self.known_face_encodings:
            self.known_face_encodings = np.asarray(self.known_face_encodings, dtype=np.float32)
        else:
            self.known_face_encodings = np.empty((0, 128), dtype=np.float32)

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
        """Load ONLY precomputed encodings (*.npy).

        This keeps startup fast and avoids importing dlib/face_recognition in the loader.
        Expected layout:
          /data/knownfaces/<PersonName>/*_opt.npy  (preferred)
        Legacy flat layout is also supported:
          /data/knownfaces/<name>_opt.npy
        """
        try:
            if not os.path.isdir(directory):
                return

            for entry in sorted(os.listdir(directory)):
                entry_path = os.path.join(directory, entry)

                # Subfolder per person
                if os.path.isdir(entry_path):
                    person_name = self._normalize_person_name(entry)
                    if not person_name:
                        continue

                    npy_files = [fn for fn in sorted(os.listdir(entry_path))
                                 if fn.lower().endswith('_opt.npy')]

                    for fn in npy_files:
                        file_path = os.path.join(entry_path, fn)
                        try:
                            enc = np.load(file_path)
                            enc = np.asarray(enc).reshape(-1)
                            if enc.shape[0] != 128:
                                continue
                            self.known_face_encodings.append(enc.astype(np.float32, copy=False))
                            self.known_face_names.append(person_name)
                        except Exception:
                            continue
                    continue

                # Legacy flat files in /knownfaces
                if entry.lower().endswith('_opt.npy') and os.path.isfile(entry_path):
                    person_name = self._name_from_filename(entry.replace('_opt.npy', ''))
                    try:
                        enc = np.load(entry_path)
                        enc = np.asarray(enc).reshape(-1)
                        if enc.shape[0] != 128:
                            continue
                        self.known_face_encodings.append(enc.astype(np.float32, copy=False))
                        self.known_face_names.append(person_name)
                    except Exception:
                        continue

        except FileNotFoundError:
            print(f"Directory {directory} not found.")
        except Exception as e:
            print(f"Failed to load known faces: {e}")


    def get_name(self, face_encoding):
        if self.known_face_encodings.shape[0] == 0:
            return "Unknown"
        # Fast squared Euclidean in float32 (avoid temporary float64 conversions)
        fe = np.asarray(face_encoding, dtype=np.float32)
        diff = self.known_face_encodings - fe
        d2 = np.einsum("ij,ij->i", diff, diff)
        best_match_index = int(np.argmin(d2))

        threshold = 0.55
        if self.config_manager is not None:
            try:
                threshold = float(self.config_manager.get('face_match_threshold', threshold))
            except Exception:
                pass

        if d2[best_match_index] < (threshold * threshold):
            return self.known_face_names[best_match_index]
        return "Unknown"
