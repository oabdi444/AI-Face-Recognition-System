import cv2
import numpy as np
from PIL import Image
import os
from deepface import DeepFace
import pandas as pd

class DeepFaceDetector:
    def __init__(self):
        self.models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]
        self.current_model = "VGG-Face"  # Default model
        
    def detect_faces(self, image):
        """Detect faces using DeepFace"""
        try:
            # Convert PIL to numpy array
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
                
            # DeepFace detect
            faces = DeepFace.extract_faces(img_array, detector_backend='opencv')
            return faces
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def recognize_faces(self, image, database_path="data/people"):
        """Recognize faces using DeepFace"""
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Find faces in database
            results = DeepFace.find(
                img_path=img_array,
                db_path=database_path,
                model_name=self.current_model,
                distance_metric="cosine",
                enforce_detection=False
            )
            
            recognition_results = []
            
            if isinstance(results, list) and len(results) > 0:
                for result_df in results:
                    if not result_df.empty:
                        # Get the best match
                        best_match = result_df.iloc[0]
                        identity = best_match['identity']
                        distance = best_match[f'{self.current_model}_cosine']
                        
                        # Extract person name from path
                        person_name = os.path.basename(os.path.dirname(identity))
                        confidence = max(0, 1 - distance)  # Convert distance to confidence
                        
                        recognition_results.append({
                            'name': person_name,
                            'confidence': confidence,
                            'distance': distance
                        })
            
            return recognition_results
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return []
    
    def verify_faces(self, img1, img2):
        """Verify if two images contain the same person"""
        try:
            result = DeepFace.verify(
                img1_path=img1,
                img2_path=img2,
                model_name=self.current_model,
                distance_metric="cosine"
            )
            return result
        except Exception as e:
            print(f"Verification error: {e}")
            return None
    
    def analyze_face(self, image):
        """Analyze face for age, gender, emotion, race"""
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
                
            analysis = DeepFace.analyze(
                img_path=img_array,
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False
            )
            return analysis
        except Exception as e:
            print(f"Analysis error: {e}")
            return None