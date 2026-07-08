import cv2
import numpy as np
import os
import time

class HumanEmotionDetector:
    def __init__(self):
        # Cascade classifiers for face, eyes, and smile detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Age and gender model lists
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']
        
        # Performance optimization - cache last detection results
        self.last_detection_time = 0
        self.detection_interval = 0.1  # Process detection every 100ms
        self.cached_boxes = []
        self.cached_emotions = []
        self.cached_ages = []
        self.cached_genders = []
        
        # Try to load pre-trained models for better accuracy
        self._load_age_gender_models()
        
        print("HumanEmotionDetector initialized with Age & Gender Detection (Optimized)!")

    def _load_age_gender_models(self):
        """Load pre-trained gender and age recognition models"""
        model_path = os.path.join(os.path.dirname(__file__), 'models')
        
        # Try to load gender net
        gender_proto = 'gender_deploy.prototxt'
        gender_model = 'gender_net.caffemodel'
        age_proto = 'age_deploy.prototxt'
        age_model = 'age_net.caffemodel'
        
        self.gender_net = None
        self.age_net_dnn = None
        
        try:
            if os.path.exists(os.path.join(model_path, gender_proto)):
                self.gender_net = cv2.dnn.readNetFromCaffe(
                    os.path.join(model_path, gender_proto),
                    os.path.join(model_path, gender_model)
                )
            if os.path.exists(os.path.join(model_path, age_proto)):
                self.age_net_dnn = cv2.dnn.readNetFromCaffe(
                    os.path.join(model_path, age_proto),
                    os.path.join(model_path, age_model)
                )
        except Exception as e:
            print(f"Note: Pre-trained models not found. Using heuristic-based detection: {e}")

    def estimate_emotion(self, face_gray):
        """Fast emotion estimation based on facial features"""
        # Simplified and faster emotion detection
        try:
            # Only check for smile (fastest)
            smile = self.smile_cascade.detectMultiScale(
                face_gray, 
                scaleFactor=1.8,  # Larger scale factor = faster
                minNeighbors=10,   # Higher threshold = fewer detections but faster
                minSize=(30, 30)
            )
            
            if len(smile) > 0:
                return "Happy"
            else:
                # Default to neutral (skip eye detection for speed)
                return "Neutral"
        except:
            return "Neutral"

    def detect_faces(self, image):
        """Detect faces and estimate emotion, age, and gender (with caching for performance)"""
        # Use cached results if detection_interval hasn't passed
        current_time = time.time()
        should_detect = (current_time - self.last_detection_time) >= self.detection_interval
        
        if should_detect:
            self.last_detection_time = current_time
            
            # Use image as-is for processing (already optimized by app.py)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use optimized Haar Cascade parameters for speed
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3,      # Larger scale = faster but less accurate
                minNeighbors=5,       # Lower threshold = faster
                minSize=(80, 80)
            )
            
            boxes = []
            emotions = []
            ages = []
            genders = []

            for (x, y, w, h) in faces:
                face_roi = image[y:y+h, x:x+w]
                
                # Only process if face is large enough
                if w > 50 and h > 50:
                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    
                    # Emotion detection (fast)
                    emotion = self.estimate_emotion(face_gray)
                    
                    # Age and Gender detection (optimized)
                    age, gender = self._estimate_age_gender_fast(face_roi)
                    
                    boxes.append((x, y, x + w, y + h))
                    emotions.append(emotion)
                    ages.append(age)
                    genders.append(gender)

            self.cached_boxes = boxes
            self.cached_emotions = emotions
            self.cached_ages = ages
            self.cached_genders = genders
        
        return self.cached_boxes, self.cached_emotions, self.cached_ages, self.cached_genders

    def _estimate_age_gender_fast(self, face_roi):
        """Fast age and gender estimation using heuristics only (no DNN)"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Quick gender estimation
            h, w = gray.shape
            left = gray[:, :w//2]
            right = cv2.flip(gray[:, w//2:], 1)
            diff = np.mean(np.abs(left - right.astype(float)))
            gender = "Male" if diff > 15 else "Female"
            
            # Quick age estimation
            texture = np.std(gray)
            if texture < 40:
                age = "(15-20)"
            elif texture < 60:
                age = "(25-32)"
            elif texture < 80:
                age = "(38-43)"
            else:
                age = "(48-60)"
            
            return age, gender
        except:
            return "(25-32)", "Male"

    def annotate_frame(self, frame):
        """Annotate frame with detected faces, emotions, age, and gender (fast)"""
        boxes, emotions, ages, genders = self.detect_faces(frame)
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            
            # Draw face rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label text
            emotion = emotions[i] if i < len(emotions) else "Unknown"
            age = ages[i] if i < len(ages) else "Unknown"
            gender = genders[i] if i < len(genders) else "Unknown"
            
            label = f"{gender} | {age} | {emotion}"
            
            # Draw label background and text (simplified for speed)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0] + 5, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame

    def shutdown(self):
        """Clean up resources"""
        print("HumanEmotionDetector shutdown complete")