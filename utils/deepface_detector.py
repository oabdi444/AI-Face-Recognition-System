# utils/deepface_detector.py

import numpy as np
from deepface import DeepFace
from deepface.commons import functions
from PIL import Image


class DeepFaceDetector:
    def __init__(self):
        self.model_name = "VGG-Face"
        self.detector_backend = "opencv"
        self.db_path = "models/known_faces"

    def recognize_faces(self, image):
        """
        Recognize faces in the given image using DeepFace's analyze.
        Returns a list of dicts with name, confidence, and bounding box.
        """
        results = []

        try:
            if isinstance(image, Image.Image):
                image = np.array(image)

            detections = DeepFace.analyze(
                img_path=image,
                actions=['identity'],
                enforce_detection=False,
                detector_backend=self.detector_backend,
                model_name=self.model_name,
                prog_bar=False
            )

            if isinstance(detections, dict):
                detections = [detections]

            for det in detections:
                name = det.get("identity", "Unknown")
                confidence = 1.0 - det.get("distance", 0.4)
                region = det.get("region", {})
                box = (
                    region.get("x", 0),
                    region.get("y", 0),
                    region.get("w", 0),
                    region.get("h", 0)
                )
                results.append({
                    "name": name.split("/")[-1].split(".")[0] if name != "Unknown" else "Unknown",
                    "confidence": confidence,
                    "box": box
                })

        except Exception as e:
            print("Recognition error:", str(e))

        return results

    def get_face_embedding(self, image):
        """
        Extract and return a face embedding from the image using DeepFace.
        This embedding can be stored in the database for future matching.
        """
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)

            face_objs = functions.extract_faces(
                img=image,
                target_size=(224, 224),
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True
            )

            if face_objs and len(face_objs) > 0:
                face_img, _ = face_objs[0]
                embedding = DeepFace.represent(
                    img_path=face_img,
                    model_name=self.model_name,
                    enforce_detection=False,
                    detector_backend=self.detector_backend
                )[0]["embedding"]

                return np.array(embedding)

        except Exception as e:
            print("Error processing image:", str(e))

        return None
