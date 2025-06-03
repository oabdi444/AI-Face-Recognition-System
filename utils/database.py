import os
import pickle
import numpy as np
from PIL import Image
from utils.deepface_detector import DeepFaceDetector  # ✅ Use DeepFaceDetector instead of face_recognition

class FaceDatabase:
    def __init__(self):
        self.db_path = "models/face_database.pkl"
        self.ensure_directories()
        self.detector = DeepFaceDetector()  # ✅ Initialize DeepFaceDetector
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs("models", exist_ok=True)
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("models/known_faces", exist_ok=True)
    
    def load_database(self):
        """Load the face database from pickle file"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def save_database(self, database):
        """Save the face database to pickle file"""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(database, f)
            return True
        except:
            return False
    
    def add_person(self, name, image_files):
        """Add a new person to the database"""
        database = self.load_database()
        encodings = []
        
        for image_file in image_files:
            try:
                # Load and process image
                image = Image.open(image_file)
                
                # Extract face embedding using DeepFaceDetector
                embedding = self.detector.get_face_embedding(image)  # ✅ Replacement
                
                if embedding is not None:
                    encodings.append(embedding)
                
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        
        if encodings:
            # Add or update person in database
            if name in database:
                database[name].extend(encodings)
            else:
                database[name] = encodings
            
            # Save database
            return self.save_database(database)
        
        return False
    
    def remove_person(self, name):
        """Remove a person from the database"""
        database = self.load_database()
        
        if name in database:
            del database[name]
            return self.save_database(database)
        
        return False
    
    def get_all_people(self):
        """Get all people in the database"""
        return self.load_database()
    
    def clear_database(self):
        """Clear the entire database"""
        return self.save_database({})
    
    def get_person_encodings(self, name):
        """Get face encodings for a specific person"""
        database = self.load_database()
        return database.get(name, [])
