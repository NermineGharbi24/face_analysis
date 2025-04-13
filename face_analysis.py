# face_analysis.py
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Union, Optional

class FaceAnalysisModel(nn.Module):
    """
    Face analysis model for detecting gender, age range, and racial characteristics
    based on the FairFace dataset approach.
    """
    def __init__(self, num_classes_gender=2, num_classes_age=9, num_classes_race=7):
        super(FaceAnalysisModel, self).__init__()
        # Use ResNet34 as the backbone
        resnet = models.resnet34(weights='DEFAULT')
        # Remove the last fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension from ResNet34
        feature_dim = 512
        
        # Task-specific heads
        self.gender_classifier = nn.Linear(feature_dim, num_classes_gender)
        self.age_classifier = nn.Linear(feature_dim, num_classes_age)
        self.race_classifier = nn.Linear(feature_dim, num_classes_race)
        
    def forward(self, x):
        features = self.backbone(x)
        # Flatten the features
        features = features.view(features.size(0), -1)
        
        # Get predictions for each task
        gender_preds = self.gender_classifier(features)
        age_preds = self.age_classifier(features)
        race_preds = self.race_classifier(features)
        
        return gender_preds, age_preds, race_preds

class FaceAnalyzer:
    """
    Complete pipeline for face detection and analysis.
    """
    def __init__(self, model_path: Optional[str] = None):
        # Load face detector (using OpenCV's DNN face detector for better performance)
        self.face_detector = cv2.dnn.readNetFromCaffe(
            'models/deploy.prototxt',
            'models/res10_300x300_ssd_iter_140000.caffemodel'
        )
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize and load the analysis model
        self.model = FaceAnalysisModel().to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Define image transforms for the model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Define class labels
        self.gender_classes = ['Male', 'Female']
        self.age_classes = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        self.race_classes = ['White', 'Black', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern', 'Latino/Hispanic']
        
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in an image using OpenCV's DNN face detector.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of face bounding boxes (x, y, w, h, confidence)
        """
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x1, y1, x2, y2 = box.astype(int)
                # Convert to x, y, w, h format
                faces.append((x1, y1, x2 - x1, y2 - y1, confidence))
                
        return faces
    
    def analyze_face(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Dict[str, str]:
        """
        Analyze a detected face for gender, age, and race.
        
        Args:
            image: Input image in BGR format
            face_box: Bounding box of the face (x, y, w, h)
            
        Returns:
            Dictionary with predictions for gender, age, and race
        """
        x, y, w, h = face_box[:4]
        
        # Extract face with a margin
        margin = 10
        x_min = max(0, x - margin)
        y_min = max(0, y - margin)
        x_max = min(image.shape[1], x + w + margin)
        y_max = min(image.shape[0], y + h + margin)
        
        # Extract and preprocess the face
        face_img = image[y_min:y_max, x_min:x_max]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)
        face_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            gender_logits, age_logits, race_logits = self.model(face_tensor)
            
            gender_pred = torch.argmax(gender_logits, dim=1).item()
            age_pred = torch.argmax(age_logits, dim=1).item()
            race_pred = torch.argmax(race_logits, dim=1).item()
            
        # Return predictions
        return {
            'gender': self.gender_classes[gender_pred],
            'age': self.age_classes[age_pred],
            'race': self.race_classes[race_pred]
        }
    
    def process_image(self, image_path: str, output_path: Optional[str] = None, show_result: bool = True) -> List[Dict]:
        """
        Process an image file for face detection and analysis.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the annotated image (optional)
            show_result: Whether to display the result
            
        Returns:
            List of dictionaries with face information and predictions
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Make a copy for visualization
        vis_image = image.copy()
        
        # Detect faces
        faces = self.detect_faces(image)
        
        results = []
        # Analyze each face
        for face_box in faces:
            x, y, w, h, conf = face_box
            
            # Analyze the face
            predictions = self.analyze_face(image, face_box)
            
            # Store results
            face_info = {
                'bbox': (x, y, w, h),
                'confidence': conf,
                'predictions': predictions
            }
            results.append(face_info)
            
            # Visualize the results
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display predictions
            label = f"{predictions['gender']}, {predictions['age']}, {predictions['race']}"
            cv2.putText(vis_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save or show the image
        if output_path:
            cv2.imwrite(output_path, vis_image)
            
        if show_result:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            
        return results
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        Process a video file for face detection and analysis.
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the processed video (optional)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec as needed
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Analyze each face
            for face_box in faces:
                x, y, w, h, conf = face_box
                
                # Analyze the face
                predictions = self.analyze_face(frame, face_box)
                
                # Visualize the results
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Display predictions
                label = f"{predictions['gender']}, {predictions['age']}, {predictions['race']}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Write frame if output is requested
            if writer:
                writer.write(frame)
                
            # Display frame (for real-time processing)
            cv2.imshow('Face Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Release resources
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
