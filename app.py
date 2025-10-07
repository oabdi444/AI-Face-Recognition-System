import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from utils.database import FaceDatabase
from utils.deepface_detector import DeepFaceDetector


class FaceRecognitionApp:
    def __init__(self):
        self.face_detector = DeepFaceDetector()
        self.database = FaceDatabase()
        
    def run(self):
        st.markdown('<h1 class="main-header"> AI Face Recognition System</h1>', unsafe_allow_html=True)
        
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a function:",
            ["Face Recognition", "Add New Person", "Manage Database", "About"]
        )
        
        if page == "Face Recognition":
            self.face_recognition_page()
        elif page == "Add New Person":
            self.add_person_page()
        elif page == "Manage Database":
            self.manage_database_page()
        elif page == "About":
            self.about_page()
    
    def face_recognition_page(self):
        st.markdown('<h2 class="section-header"> Face Recognition</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image containing faces to recognize"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button(" Recognize Faces", type="primary"):
                    with st.spinner("Processing image..."):
                        results = self.face_detector.recognize_faces(image)
                        if results:
                            st.success(f"Found {len(results)} face(s)!")
                            with col2:
                                st.subheader("Recognition Results")
                                for i, result in enumerate(results):
                                    st.write(f"**Face {i+1}:**")
                                    st.write(f"- Name: {result['name']}")
                                    st.write(f"- Confidence: {result['confidence']:.2%}")
                                    st.write("---")
                        else:
                            st.error("No faces detected in the image.")
        
        with col2:
            st.subheader("Live Camera Feed")
            if st.button(" Start Camera Recognition"):
                self.camera_recognition()
    
    def add_person_page(self):
        st.markdown('<h2 class="section-header">➕ Add New Person</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Person Information")
            name = st.text_input("Enter person's name:", placeholder="John Doe")
            
            st.subheader("Upload Photos")
            uploaded_files = st.file_uploader(
                "Choose image files",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                help="Upload multiple clear photos of the person's face"
            )
            
            if uploaded_files and name:
                st.write(f"Selected {len(uploaded_files)} image(s)")
                cols = st.columns(min(len(uploaded_files), 3))
                for i, file in enumerate(uploaded_files[:3]):
                    with cols[i % 3]:
                        image = Image.open(file)
                        st.image(image, caption=f"Photo {i+1}", use_column_width=True)
                
                if st.button(" Add Person to Database", type="primary"):
                    with st.spinner("Processing and saving..."):
                        # Convert uploaded_files to list of PIL Images or file objects compatible with database
                        # Your FaceDatabase.add_person expects list of files or file-like objects, so pass as is
                        success = self.database.add_person(name, uploaded_files)
                        
                        if success:
                            st.markdown(
                                f'<div class="success-box"> Successfully added {name} to the database!</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<div class="error-box"> Failed to add person. Please ensure faces are clearly visible.</div>',
                                unsafe_allow_html=True
                            )
        
        with col2:
            st.subheader("Tips for Best Results")
            st.info("""
             **For optimal face recognition:**
            
            • Upload 3-5 clear photos of the person  
            • Ensure good lighting in photos  
            • Face should be clearly visible and unobstructed  
            • Include photos from different angles  
            • Avoid blurry or low-quality images  
            • One person per photo works best
            """)
    
    def manage_database_page(self):
        st.markdown('<h2 class="section-header"> Manage Database</h2>', unsafe_allow_html=True)
        people = self.database.get_all_people()
        
        if people:
            st.subheader(f"Database contains {len(people)} person(s)")
            df = pd.DataFrame(list(people.items()), columns=['Name', 'Encodings'])
            df['Encodings Count'] = df['Encodings'].apply(len)
            df = df.drop(columns=['Encodings'])
            st.dataframe(df, use_container_width=True)
            
            st.subheader("Remove Person")
            person_to_delete = st.selectbox("Select person to remove:", list(people.keys()))
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if st.button("🗑️ Delete", key="delete"):
                    if self.database.remove_person(person_to_delete):
                        st.success(f"Removed {person_to_delete} from database")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to remove person")
            
            with col2:
                if st.button(" Clear All Data", key="clear"):
                    if st.confirm("Are you sure you want to clear all data?"):
                        self.database.clear_database()
                        st.success("Database cleared!")
                        st.experimental_rerun()
        else:
            st.info("Database is empty. Add some people first!")
    
    def about_page(self):
        st.markdown('<h2 class="section-header"> About This App</h2>', unsafe_allow_html=True)
        st.markdown("""
        ##  AI Face Recognition System
        
    
    def camera_recognition(self):
        st.info("Camera recognition feature - implement based on your camera setup")
        # Placeholder for future camera implementation

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()
