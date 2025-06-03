import os
import pickle
from utils.deepface_detector import DeepFaceDetector
from PIL import Image

class FaceDatabase:
    def __init__(self):
        self.db_path = "models/face_database.pkl"
        self.ensure_directories()
        self.face_detector = DeepFaceDetector()  # Initialize detector here

    def ensure_directories(self):
        os.makedirs("models", exist_ok=True)
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("models/known_faces", exist_ok=True)

    def load_database(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}

    def save_database(self, database):
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(database, f)
            return True
        except:
            return False

    def add_person(self, name, image_files):
        database = self.load_database()
        encodings = []

        for image_file in image_files:
            try:
                image = Image.open(image_file)
                # Use DeepFaceDetector to get face embedding(s)
                embedding = self.face_detector.get_face_embedding(image)
                if embedding is not None:
                    encodings.append(embedding)
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        
        if encodings:
            if name in database:
                database[name].extend(encodings)
            else:
                database[name] = encodings
            return self.save_database(database)
        
        return False

    # Add your other existing methods here if you want (remove_person, get_all_people, etc.)
