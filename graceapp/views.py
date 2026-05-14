import os
import json
import uuid
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from math import sqrt, acos
from typing import List, Tuple, Dict
from transformers import DeiTForImageClassification
import torch.nn as nn
from ultralytics import YOLO

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .models import UserImage, FashionRecommendation

# ==================== FACE SHAPE DETECTION CLASS ====================

class FaceShapeDetector:
    def __init__(self):
        """Initialize MediaPipe Face Mesh and required components"""
        self.face_mesh = None
        self.mp_face_mesh = None
        
        try:
            # Try old API first (MediaPipe < 0.10)
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
        except (AttributeError, ImportError) as e:
            # New API (MediaPipe >= 0.10) or not installed - will handle gracefully
            print(f"MediaPipe Face Mesh not available: {e}")
            self.face_mesh = None
        
        # Define facial landmark indices for shape analysis
        self.LANDMARK_INDICES = {
            'jaw': list(range(0, 17)),
            'face_oval': list(range(10, 151))
        }
    
    def detect_landmarks(self, image: np.ndarray) -> List:
        """Detect facial landmarks using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        return results.multi_face_landmarks[0]
    
    def calculate_face_metrics(self, landmarks, image_shape: Tuple[int, int]) -> Dict:
        """Calculate various facial measurements for shape classification"""
        height, width = image_shape[:2]
        
        # Get normalized landmark coordinates
        landmark_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmark_points.append((x, y))
        
        # Key points indices (MediaPipe 468 landmarks)
        jaw_left = landmark_points[172]   # Left jaw
        jaw_right = landmark_points[397]  # Right jaw
        chin = landmark_points[152]       # Chin
        forehead = landmark_points[10]    # Forehead center
        cheek_left = landmark_points[123]
        cheek_right = landmark_points[352]
        
        # Face width measurements
        face_width = self._euclidean_distance(jaw_left, jaw_right)
        jaw_width = self._euclidean_distance(landmark_points[58], landmark_points[288])
        cheekbone_width = self._euclidean_distance(cheek_left, cheek_right)
        forehead_width = self._euclidean_distance(landmark_points[54], landmark_points[284])
        
        # Face height measurements
        face_height = self._euclidean_distance(forehead, chin)
        
        # Ratios for classification
        face_ratio = face_height / face_width if face_width > 0 else 1.0
        jaw_face_ratio = jaw_width / face_width if face_width > 0 else 0.0
        cheekbone_face_ratio = cheekbone_width / face_width if face_width > 0 else 0.0
        forehead_face_ratio = forehead_width / face_width if face_width > 0 else 0.0
        
        # Calculate face angles
        jaw_angle = self._calculate_jaw_angle(landmark_points)
        
        # Calculate facial contours
        jaw_points = [landmark_points[i] for i in range(0, 17)]
        face_oval_points = [landmark_points[i] for i in self.LANDMARK_INDICES['face_oval']]
        
        # Calculate symmetry
        symmetry_score = self._calculate_face_symmetry(landmark_points, width)
        
        return {
            'face_width': face_width,
            'face_height': face_height,
            'jaw_width': jaw_width,
            'cheekbone_width': cheekbone_width,
            'forehead_width': forehead_width,
            'face_ratio': face_ratio,
            'jaw_face_ratio': jaw_face_ratio,
            'cheekbone_face_ratio': cheekbone_face_ratio,
            'forehead_face_ratio': forehead_face_ratio,
            'jaw_angle': jaw_angle,
            'symmetry_score': symmetry_score,
            'landmark_points': landmark_points,
            'jaw_points': jaw_points,
            'face_oval_points': face_oval_points
        }
    
    def classify_face_shape(self, metrics: Dict) -> Tuple[str, float]:
        """Classify face shape based on calculated metrics - BASIC SHAPES ONLY"""
        
        face_ratio = metrics['face_ratio']
        jaw_ratio = metrics['jaw_face_ratio']
        cheekbone_ratio = metrics['cheekbone_face_ratio']
        forehead_ratio = metrics['forehead_face_ratio']
        
        # Initialize scores for basic face shapes only
        shape_scores = {
            'Oval': 0,
            'Round': 0,
            'Square': 0,
            'Heart': 0
        }
        
        # Simplified rule-based scoring system
        if 1.3 <= face_ratio <= 1.5:
            shape_scores['Oval'] += 3
        if abs(cheekbone_ratio - jaw_ratio) < 0.1:
            shape_scores['Oval'] += 2
        
        if 0.9 <= face_ratio <= 1.1:
            shape_scores['Round'] += 3
        if cheekbone_ratio > 0.9:
            shape_scores['Round'] += 2
        
        if jaw_ratio > 0.85:
            shape_scores['Square'] += 3
        
        if forehead_ratio > cheekbone_ratio + 0.1:
            shape_scores['Heart'] += 3
        if jaw_ratio < cheekbone_ratio - 0.1:
            shape_scores['Heart'] += 2
        
        # Determine the face shape with highest score
        face_shape = max(shape_scores, key=shape_scores.get)
        
        # Calculate confidence
        total_score = sum(shape_scores.values())
        confidence = shape_scores[face_shape] / total_score if total_score > 0 else 0
        
        return face_shape, confidence
    
    def _euclidean_distance(self, point1: Tuple, point2: Tuple) -> float:
        """Calculate Euclidean distance between two points"""
        return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _calculate_jaw_angle(self, landmark_points: List) -> float:
        """Calculate jaw angle from jawline points"""
        jaw_points = [landmark_points[i] for i in [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288]]
        
        if len(jaw_points) < 3:
            return 90.0
        
        angles = []
        for i in range(1, len(jaw_points) - 1):
            v1 = (jaw_points[i-1][0] - jaw_points[i][0], jaw_points[i-1][1] - jaw_points[i][1])
            v2 = (jaw_points[i+1][0] - jaw_points[i][0], jaw_points[i+1][1] - jaw_points[i][1])
            
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag_v1 = sqrt(v1[0]**2 + v1[1]**2)
            mag_v2 = sqrt(v2[0]**2 + v2[1]**2)
            
            if mag_v1 * mag_v2 > 0:
                angle = np.degrees(acos(min(max(dot_product/(mag_v1*mag_v2), -1), 1)))
                angles.append(angle)
        
        return np.mean(angles) if angles else 90.0
    
    def _calculate_face_symmetry(self, landmark_points: List, image_width: int) -> float:
        """Calculate face symmetry score (0-1)"""
        symmetry_pairs = [
            (93, 323), (130, 359), (50, 280), (61, 291)
        ]
        
        left_points = []
        right_points = []
        
        for left_idx, right_idx in symmetry_pairs:
            if left_idx < len(landmark_points) and right_idx < len(landmark_points):
                left_points.append(landmark_points[left_idx])
                right_points.append(landmark_points[right_idx])
        
        if not left_points or not right_points:
            return 0.5
        
        distances = []
        for left_pt, right_pt in zip(left_points, right_points):
            mirrored_right_x = image_width - right_pt[0]
            distance = self._euclidean_distance(left_pt, (mirrored_right_x, right_pt[1]))
            distances.append(distance)
        
        avg_distance = np.mean(distances)
        max_possible_distance = image_width / 4
        symmetry = 1.0 - min(avg_distance / max_possible_distance, 1.0)
        
        return symmetry

# ==================== SKIN TONE DETECTION MODEL ====================

import timm

class SkinToneDetector:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device for skin tone: {self.device}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint["class_names"]
        
        self.model = timm.create_model(
            "deit_small_patch16_224", 
            pretrained=False, 
            num_classes=len(self.class_names)
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def predict(self, image_path):
        """Predict skin tone from image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(img_tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(prob, 1)
            
            predicted_class = self.class_names[predicted_idx.item()]
            confidence = confidence.item()
            
            return {
                'skin_tone': predicted_class,
                'confidence': confidence,
                'all_classes': self.class_names
            }
        except Exception as e:
            print(f"Error in skin tone prediction: {e}")
            return None

# ==================== BODY SHAPE DETECTION MODEL ====================

class BodyShapeDetector:
    def __init__(self, model_path):
        print("Loading YOLO model for body shape detection...")
        self.model = YOLO(model_path)
        self.class_names = {0: 'apple', 1: 'hourglass', 2: 'inverted_triangle', 3: 'pear', 4: 'rectangle'}
    
    def predict(self, image_path):
        """Predict body shape from image"""
        try:
            results = self.model(
                source=image_path,
                imgsz=640,
                save=False,
            )
            
            result = results[0]
            if result.probs is not None:
                top1_idx = result.probs.top1
                top1_conf = float(result.probs.top1conf)
                class_name = self.class_names[top1_idx]
                
                return {
                    'body_shape': class_name,
                    'confidence': top1_conf
                }
            else:
                return None
        except Exception as e:
            print(f"Error in body shape prediction: {e}")
            return None

# ==================== RULE-BASED RECOMMENDATION ENGINE  ====================

class FashionRuleEngine:
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize all fashion rules based on skin tone, body shape, and face shape"""
        rules = {}
        
        # COLOR PALETTE RECOMMENDATIONS
        color_rules = {
            'light': {
                'best_colors': ['Pastels', 'Soft neutrals', 'Light blues', 'Lavender', 'Mint green', 'Peach'],
                'avoid': ['Very dark colors without contrast', 'Neon colors'],
                'patterns': ['Subtle floral prints', 'Delicate patterns', 'Small geometric designs'],
                'fabrics': ['Silk', 'Chiffon', 'Linen', 'Light cotton']
            },
            'mid-light': {
                'best_colors': ['Medium blues', 'Sage green', 'Dusty rose', 'Mauve', 'Warm beige', 'Olive'],
                'avoid': ['Very pale washed out colors', 'Bright yellow'],
                'patterns': ['Medium contrast patterns', 'Watercolor prints', 'Abstract designs'],
                'fabrics': ['Cotton', 'Linen', 'Light wool', 'Viscose']
            },
            'mid-dark': {
                'best_colors': ['Earth tones', 'Rich reds', 'Burnt orange', 'Gold', 'Emerald green', 'Royal blue'],
                'avoid': ['Very light pastels without contrast', 'Muddy colors'],
                'patterns': ['Ethnic prints', 'Warm-toned patterns', 'Bold geometrics'],
                'fabrics': ['Silk', 'Velvet', 'Cotton blends', 'Lightweight wool']
            },
            'dark': {
                'best_colors': ['Bright jewel tones', 'Deep reds', 'Electric blue', 'Violet', 'Fuchsia', 'Metallics'],
                'avoid': ['Very dark browns and blacks without contrast', 'Dull colors'],
                'patterns': ['Bold prints', 'High contrast patterns', 'African prints'],
                'fabrics': ['Satin', 'Velvet', 'Rich cottons', 'Brocade']
            }
        }
        
        # GENDER-SEPARATE BODY SHAPE GUIDELINES - NO SHARED TERMS
        self.male_shape_rules = {
            'apple': {
                'fit': ['Structured jackets', 'V-neck shirts', 'Vertical stripes', 'Straight-cut trousers'],
                'avoid': ['Tight waistbands', 'Cropped tops', 'Skinny fits around midsection'],
                'silhouette': ['Create clean lines from shoulder to hip', 'Draw attention to shoulders and legs'],
                'accessories': ['Watches', 'Ties', 'Pocket squares', 'Belts']
            },
            'rectangle': {
                'fit': ['Layered looks', 'Textured fabrics', 'Patterned shirts', 'Structured blazers'],
                'avoid': ['Overly boxy cuts', 'Shapeless oversized garments'],
                'silhouette': ['Add visual interest through layering', 'Create shoulder definition'],
                'accessories': ['Statement watches', 'Leather bracelets', 'Lapel pins']
            },
            'inverted_triangle': {
                'fit': ['Straight-leg trousers', 'Light colored bottoms', 'V-neck tops', 'Unstructured jackets'],
                'avoid': ['Broad shoulder pads', 'Heavy upper body details'],
                'silhouette': ['Balance broad shoulders with lower body', 'Add volume to legs'],
                'accessories': ['Hip-level belts', 'Statement socks', 'Sleek shoes']
            },
            'pear': {
                'fit': ['Structured shoulder jackets', 'Dark trousers', 'Tailored shirts', 'Bold upper body patterns'],
                'avoid': ['Tight trousers', 'Light colored pants', 'Skinny fits'],
                'silhouette': ['Broaden shoulder appearance', 'Streamline lower body'],
                'accessories': ['Wide collar details', 'Shoulder accents', 'Statement watches']
            },
            'hourglass': {
                'fit': ['Fitted suits', 'Tailored shirts', 'Tapered trousers', 'Defined waist jackets'],
                'avoid': ['Baggy oversized fits', 'Shapeless garments'],
                'silhouette': ['Highlight natural proportions', 'Maintain balanced silhouette'],
                'accessories': ['Slim belts', 'Tailored watches', 'Classic ties']
            }
        }
        
        self.female_shape_rules = {
            'apple': {
                'fit': ['Empire waist dresses', 'A-line silhouettes', 'V-neck tops', 'Wrap dresses'],
                'avoid': ['Tight waistbands', 'Crop tops', 'High-waisted pants'],
                'silhouette': ['Create definition below bust', 'Draw attention to legs and neckline'],
                'accessories': ['Statement necklaces', 'Long pendants', 'Drop earrings']
            },
            'rectangle': {
                'fit': ['Belted dresses', 'Peplum tops', 'Ruffled details', 'Layered looks'],
                'avoid': ['Boxy silhouettes', 'Shapeless garments'],
                'silhouette': ['Create waist definition', 'Add feminine curves'],
                'accessories': ['Waist belts', 'Hoop earrings', 'Statement bracelets']
            },
            'inverted_triangle': {
                'fit': ['A-line skirts', 'Full skirts', 'Wide-leg pants', 'Soft draped tops'],
                'avoid': ['Broad shoulder details', 'Puffed sleeves', 'Shoulder pads'],
                'silhouette': ['Balance upper and lower body', 'Add volume to hips'],
                'accessories': ['Hip belts', 'Statement shoes', 'Lower body jewelry']
            },
            'pear': {
                'fit': ['Bright patterned tops', 'Dark bottoms', 'Off-shoulder styles', 'Structured jackets'],
                'avoid': ['Tight bottoms', 'Light colored pants', 'Bodycon skirts'],
                'silhouette': ['Balance shoulders with hips', 'Draw attention upward'],
                'accessories': ['Statement earrings', 'Neck scarves', 'Bold necklaces']
            },
            'hourglass': {
                'fit': ['Fitted dresses', 'Wrap styles', 'Belted waists', 'Bodycon silhouettes'],
                'avoid': ['Shapeless garments', 'Oversized boxy styles'],
                'silhouette': ['Emphasize natural waist', 'Showcase feminine curves'],
                'accessories': ['Waist belts', 'Delicate necklaces', 'Chic bracelets']
            }
        }
        
        # FACE SHAPE GUIDELINES - BASIC SHAPES ONLY
        face_rules = {
            'Oval': {
                'necklines': ['All necklines work well', 'V-neck', 'Round neck', 'Square neck'],
                'avoid': ['Extreme necklines that disrupt balance'],
                'hair_accessories': ['All styles work', 'Side parts', 'Center parts'],
                'glasses': ['Oval frames', 'Rectangular frames', 'Wayfarer style']
            },
            'Round': {
                'necklines': ['V-neck', 'Deep V-neck', 'Square neck', 'Off-shoulder'],
                'avoid': ['Round necklines', 'High necklines'],
                'hair_accessories': ['Angular hairstyles', 'High ponytails', 'Side-swept bangs'],
                'glasses': ['Angular frames', 'Rectangular frames', 'Cat-eye']
            },
            'Square': {
                'necklines': ['Round neck', 'V-neck', 'Scoop neck', 'Off-shoulder'],
                'avoid': ['Square necklines', 'Straight necklines'],
                'hair_accessories': ['Soft waves', 'Layered hairstyles', 'Side parts'],
                'glasses': ['Round frames', 'Oval frames', 'Aviators']
            },
            'Heart': {
                'necklines': ['V-neck', 'Scoop neck', 'Sweetheart neckline', 'Off-shoulder'],
                'avoid': ['High necklines', 'Turtle necks'],
                'hair_accessories': ['Side parts', 'Wispy bangs', 'Chin-length bobs'],
                'glasses': ['Round bottom frames', 'Aviators', 'Light-colored frames']
            }
        }
        
        # GENDER-SEPARATE STYLE RECOMMENDATIONS - NO SHARED TERMS
        self.male_style_rules = {
            # APPLE BODY SHAPE
            'apple_formal': {'description': 'Structured formal for apple', 'best_for_skin': ['all'], 'best_for_body': ['apple'], 'elements': ['Single-breasted suits', 'Vertical stripes', 'V-neck sweaters', 'Dark trousers'], 'footwear': ['Oxford shoes', 'Derby shoes']},
            'apple_casual': {'description': 'Casual for apple', 'best_for_skin': ['all'], 'best_for_body': ['apple'], 'elements': ['Untucked shirts', 'Straight-leg jeans', 'Lightweight jackets'], 'footwear': ['Clean sneakers', 'Loafers']},
            'apple_street': {'description': 'Urban for apple', 'best_for_skin': ['all'], 'best_for_body': ['apple'], 'elements': ['Long-line tees', 'Slim joggers', 'Bomber jackets'], 'footwear': ['High-top sneakers', 'Running shoes']},
            # RECTANGLE
            'rectangle_formal': {'description': 'Layered formal for rectangle', 'best_for_skin': ['all'], 'best_for_body': ['rectangle'], 'elements': ['Three-piece suits', 'Textured blazers', 'Waistcoats'], 'footwear': ['Brogues', 'Leather boots']},
            'rectangle_casual': {'description': 'Textured casual for rectangle', 'best_for_skin': ['all'], 'best_for_body': ['rectangle'], 'elements': ['Cable knits', 'Quilted jackets', 'Patterned shirts'], 'footwear': ['Desert boots', 'Chukka boots']},
            'rectangle_smart': {'description': 'Smart for rectangle', 'best_for_skin': ['all'], 'best_for_body': ['rectangle'], 'elements': ['Double-breasted blazers', 'Contrast collars', 'Pocket squares'], 'footwear': ['Loafers', 'Penny loafers']},
            # INVERTED TRIANGLE
            'inverted_formal': {'description': 'Balanced formal for inverted', 'best_for_skin': ['all'], 'best_for_body': ['inverted_triangle'], 'elements': ['Unstructured blazers', 'Light trousers', 'Simple shirts'], 'footwear': ['Sleek loafers', 'Light leather']},
            'inverted_casual': {'description': 'Lower-body focus for inverted', 'best_for_skin': ['all'], 'best_for_body': ['inverted_triangle'], 'elements': ['Bootcut jeans', 'Cargo pants', 'Light wash denim'], 'footwear': ['White sneakers', 'Tan boots']},
            'inverted_athleisure': {'description': 'Sporty for inverted', 'best_for_skin': ['all'], 'best_for_body': ['inverted_triangle'], 'elements': ['Track pants', 'Athletic joggers', 'Performance polos'], 'footwear': ['Running shoes', 'Cross-trainers']},
            # PEAR
            'pear_formal': {'description': 'Shoulder-enhancing for pear', 'best_for_skin': ['all'], 'best_for_body': ['pear'], 'elements': ['Wide lapels', 'Structured shoulders', 'Patterned shirts'], 'footwear': ['Pointed oxfords', 'Dress boots']},
            'pear_casual': {'description': 'Upper focus for pear', 'best_for_skin': ['all'], 'best_for_body': ['pear'], 'elements': ['Epaulette jackets', 'Wide collars', 'Horizontal stripes'], 'footwear': ['Pointed shoes', 'Chelsea boots']},
            'pear_classic': {'description': 'Classic for pear', 'best_for_skin': ['all'], 'best_for_body': ['pear'], 'elements': ['Trench coats', 'Shoulder details', 'Bold patterns'], 'footwear': ['Brogues', 'Chelsea boots']},
            # HOURGLASS
            'hourglass_formal': {'description': 'Fitted formal for hourglass', 'best_for_skin': ['all'], 'best_for_body': ['hourglass'], 'elements': ['Slim suits', 'Fitted shirts', 'Tapered trousers'], 'footwear': ['Polished oxfords', 'Refined loafers']},
            'hourglass_smart': {'description': 'Smart for hourglass', 'best_for_skin': ['all'], 'best_for_body': ['hourglass'], 'elements': ['Slim chinos', 'Fitted polos', 'Tailored blazers'], 'footwear': ['Chelsea boots', 'Minimalist sneakers']},
            'hourglass_classic': {'description': 'Classic for hourglass', 'best_for_skin': ['all'], 'best_for_body': ['hourglass'], 'elements': ['Fitted trenches', 'Wool coats', 'Slim trousers'], 'footwear': ['Brogues', 'Leather boots']}
        }

        self.female_style_rules = {
            # APPLE
            'apple_elegant': {'description': 'Elegant for apple', 'best_for_skin': ['all'], 'best_for_body': ['apple'], 'elements': ['Empire waist', 'A-line skirts', 'Flowy blouses'], 'footwear': ['Kitten heels', 'Wedges']},
            'apple_professional': {'description': 'Professional for apple', 'best_for_skin': ['all'], 'best_for_body': ['apple'], 'elements': ['Structured blazers', 'Straight trousers', 'Tunic tops'], 'footwear': ['Pumps', 'Low heels']},
            'apple_casual': {'description': 'Casual for apple', 'best_for_skin': ['all'], 'best_for_body': ['apple'], 'elements': ['Swing dresses', 'Long cardigans', 'Bootcut jeans'], 'footwear': ['Fashion sneakers', 'Ankle boots']},
            # RECTANGLE
            'rectangle_curves': {'description': 'Curves for rectangle', 'best_for_skin': ['all'], 'best_for_body': ['rectangle'], 'elements': ['Peplum tops', 'Belted dresses', 'Ruffled blouses'], 'footwear': ['Strappy heels', 'Platform pumps']},
            'rectangle_layered': {'description': 'Layered for rectangle', 'best_for_skin': ['all'], 'best_for_body': ['rectangle'], 'elements': ['Cropped jackets', 'High-waisted skirts', 'Textured sweaters'], 'footwear': ['Ankle boots', 'Ballet flats']},
            'rectangle_feminine': {'description': 'Feminine for rectangle', 'best_for_skin': ['all'], 'best_for_body': ['rectangle'], 'elements': ['Bow blouses', 'Pleated skirts', 'Cinched waists'], 'footwear': ['Delicate heels', 'Pointed flats']},
            # INVERTED
            'inverted_balance': {'description': 'Balanced for inverted', 'best_for_skin': ['all'], 'best_for_body': ['inverted_triangle'], 'elements': ['A-line dresses', 'Full skirts', 'Wide-leg pants'], 'footwear': ['Ballet flats', 'Feminine oxfords']},
            'inverted_volume': {'description': 'Volume for inverted', 'best_for_skin': ['all'], 'best_for_body': ['inverted_triangle'], 'elements': ['Tiered skirts', 'Balloon pants', 'Patterned bottoms'], 'footwear': ['Sleek loafers', 'Minimalist heels']},
            'inverted_soft': {'description': 'Soft for inverted', 'best_for_skin': ['all'], 'best_for_body': ['inverted_triangle'], 'elements': ['Raglan sleeves', 'Boat necks', 'Soft knits'], 'footwear': ['Light footwear', 'Feminine slides']},
            # PEAR
            'pear_statement': {'description': 'Statement for pear', 'best_for_skin': ['all'], 'best_for_body': ['pear'], 'elements': ['Off-shoulder tops', 'Bright blouses', 'Statement sleeves'], 'footwear': ['Pointed pumps', 'High heels']},
            'pear_dark_bottoms': {'description': 'Dark bottoms for pear', 'best_for_skin': ['all'], 'best_for_body': ['pear'], 'elements': ['Dark jeans', 'Black trousers', 'Navy skirts'], 'footwear': ['Statement heels', 'Platform shoes']},
            'pear_balance': {'description': 'Balance for pear', 'best_for_skin': ['all'], 'best_for_body': ['pear'], 'elements': ['Fit-and-flare', 'Structured shoulders', 'Empire waists'], 'footwear': ['Elegant pumps', 'Strappy sandals']},
            # HOURGLASS
            'hourglass_fitted': {'description': 'Fitted for hourglass', 'best_for_skin': ['all'], 'best_for_body': ['hourglass'], 'elements': ['Bodycon dresses', 'Fitted blazers', 'Pencil skirts'], 'footwear': ['Classic pumps', 'Slingback heels']},
            'hourglass_belted': {'description': 'Belted for hourglass', 'best_for_skin': ['all'], 'best_for_body': ['hourglass'], 'elements': ['Belted coats', 'Cinched waists', 'High-waisted pants'], 'footwear': ['Ankle strap heels', 'Classic pumps']},
            'hourglass_curves': {'description': 'Curves for hourglass', 'best_for_skin': ['all'], 'best_for_body': ['hourglass'], 'elements': ['Sweetheart necklines', 'Mermaid skirts', 'Fitted sheaths'], 'footwear': ['Strappy heels', 'Feminine sandals']}
        }

        # GENDER-SEPARATE FOOTWEAR RECOMMENDATIONS - NO SHARED TERMS
        self.male_footwear_rules = {
            'apple': {
                'recommended': ['Oxford shoes', 'Chelsea boots', 'Minimalist leather sneakers', 'Monk strap shoes'],
                'avoid': ['Bulky running shoes', 'Chunky skate shoes'],
                'tips': ['Create clean leg lines', 'Choose streamlined profiles']
            },
            'rectangle': {
                'recommended': ['Desert boots', 'Chukka boots', 'Classic loafers', 'Brogue shoes'],
                'avoid': ['Overly minimal designs', 'Plain canvas sneakers'],
                'tips': ['Add texture with shoe details', 'Choose substantial footwear']
            },
            'inverted_triangle': {
                'recommended': ['Sleek loafers', 'Minimalist sneakers', 'Light leather shoes', 'Clean white sneakers'],
                'avoid': ['Heavy work boots', 'Chunky hiking shoes'],
                'tips': ['Keep footwear lightweight', 'Avoid overly bulky silhouettes']
            },
            'pear': {
                'recommended': ['Pointed dress shoes', 'Sleek oxford shoes', 'Tapered boots', 'Elegant loafers'],
                'avoid': ['Round toe clown shoes', 'Wide skate shoes'],
                'tips': ['Lengthen leg appearance', 'Choose refined toe shapes']
            },
            'hourglass': {
                'recommended': ['Classic oxford shoes', 'Tailored boots', 'Refined loafers', 'Elegant brogues'],
                'avoid': ['Casual flip flops', 'Bulky athletic shoes'],
                'tips': ['Maintain balanced proportions', 'Choose polished formal styles']
            }
        }
        
        self.female_footwear_rules = {
            'apple': {
                'recommended': ['Pointed toe pumps', 'Ankle strap heels', 'Kitten heels', 'Elegant wedges'],
                'avoid': ['Chunky platform sandals', 'Ballet flats', 'Round toe shoes'],
                'tips': ['Create leg elongation', 'Balance upper body with sleek footwear']
            },
            'rectangle': {
                'recommended': ['Strappy high heels', 'Platform pumps', 'Statement ankle boots', 'Dressy booties'],
                'avoid': ['Simple plain flats', 'Minimalist boring styles'],
                'tips': ['Add feminine curves with shoe shape', 'Create visual interest with details']
            },
            'inverted_triangle': {
                'recommended': ['Delicate ballet flats', 'Feminine oxfords', 'Sleek loafers', 'Simple elegant pumps'],
                'avoid': ['Heavy combat boots', 'Chunky platform shoes'],
                'tips': ['Keep footwear feminine and light', 'Balance broad shoulders with delicate shoes']
            },
            'pear': {
                'recommended': ['Pointed stiletto pumps', 'High heels', 'Ankle boots', 'Platform heels'],
                'avoid': ['Flat sandals', 'Round toe flats', 'Chunky sneakers'],
                'tips': ['Lengthen legs with heels', 'Draw attention upward with statement footwear']
            },
            'hourglass': {
                'recommended': ['Classic elegant pumps', 'Strappy high heels', 'Dressy ankle boots', 'Slingback heels'],
                'avoid': ['Chunky casual styles', 'Overly sporty shoes'],
                'tips': ['Emphasize balanced feminine proportions', 'Choose elegant sophisticated styles']
            }
        }
        
        # COMBINATION RULES (Skin Tone x Body Shape x Face Shape)
        skin_tones = ['light', 'mid-light', 'mid-dark', 'dark']
        body_shapes = ['apple', 'rectangle', 'inverted_triangle', 'pear', 'hourglass']
        face_shapes = ['Oval', 'Round', 'Square', 'Heart']
        
        for skin_tone in skin_tones:
            for body_shape in body_shapes:
                for face_shape in face_shapes:
                    key = f"{skin_tone}_{body_shape}_{face_shape}"
                    
                    # Determine best pattern styles - use female as default for initial rules
                    recommended_styles = self._get_recommended_styles(skin_tone, body_shape, 'female')
                    
                    # Generate personalized recommendations
                    recommendations = {
                        'skin_tone': skin_tone,
                        'body_shape': body_shape,
                        'face_shape': face_shape,
                        'color_palette': color_rules[skin_tone]['best_colors'],
                        'avoid_colors': color_rules[skin_tone]['avoid'],
                        'recommended_patterns': color_rules[skin_tone]['patterns'],
                        'recommended_fabrics': color_rules[skin_tone]['fabrics'],
                        'recommended_styles': recommended_styles
                    }
                    
                    rules[key] = recommendations
        
        return rules
    
    def _get_recommended_styles(self, skin_tone, body_shape, gender='female'):
        """Get recommended pattern/styles based on skin tone, body shape, and gender"""
        # Select gender-specific style rules
        style_rules = self.male_style_rules if gender == 'male' else self.female_style_rules
        
        recommended = []
        
        for style_name, style_info in style_rules.items():
            skin_match = skin_tone in style_info['best_for_skin'] or 'all' in style_info['best_for_skin']
            body_match = body_shape in style_info['best_for_body'] or 'all' in style_info['best_for_body']
            
            if skin_match and body_match:
                recommended.append({
                    'name': style_name,
                    'description': style_info.get('description', ''),
                    'elements': style_info.get('elements', []),
                    'footwear': style_info.get('footwear', [])
                })

        # Always include at least 3 styles, fallback to any available styles
        if len(recommended) < 3:
            for style_name, style_info in style_rules.items():
                if style_name not in [r['name'] for r in recommended]:
                    recommended.append({
                        'name': style_name,
                        'description': style_info.get('description', ''),
                        'elements': style_info.get('elements', []),
                        'footwear': style_info.get('footwear', [])
                    })
                    if len(recommended) >= 3:
                        break
        
        return recommended[:4]  # Return top 4 styles

    def _generate_outfit_examples(self, skin_tone, body_shape, face_shape, recommended_styles, gender='female'):
        """Generate gender-specific outfit examples"""
        examples = []
        
        if gender == 'male':
            # Male-specific outfit examples
            if body_shape == 'apple':
                examples.extend([
                    "Dark tailored blazer with straight-leg trousers",
                    "Vertical striped shirt with dark chinos",
                    "Structured polo with relaxed-fit jeans"
                ])
            elif body_shape == 'rectangle':
                examples.extend([
                    "Layered look with textured blazer",
                    "Patterned shirt with slim-fit trousers",
                    "Denim jacket with fitted t-shirt"
                ])
            elif body_shape == 'inverted_triangle':
                examples.extend([
                    "Light colored trousers with fitted polo",
                    "Straight-leg jeans with simple t-shirt",
                    "Chinos with relaxed-fit button-up"
                ])
            elif body_shape == 'pear':
                examples.extend([
                    "Structured blazer with dark trousers",
                    "Patterned shirt with straight-leg pants",
                    "Light top with tailored dark jeans"
                ])
            elif body_shape == 'hourglass':
                examples.extend([
                    "Fitted blazer with tapered trousers",
                    "Tailored shirt with slim-fit chinos",
                    "V-neck sweater with fitted jeans"
                ])
        else:
            # Female-specific outfit examples (original)
            if body_shape == 'apple':
                examples.extend([
                    "A-line dress with V-neck",
                    "Wrap dress with flattering colors",
                    "Dark jeans with flowy top"
                ])
            elif body_shape == 'rectangle':
                examples.extend([
                    "Belted dress with defined waist",
                    "Peplum top with fitted skirt",
                    "Layered look with accessories"
                ])
            elif body_shape == 'inverted_triangle':
                examples.extend([
                    "A-line skirt with fitted top",
                    "V-neck top with flared pants",
                    "Dark top with patterned skirt"
                ])
            elif body_shape == 'pear':
                examples.extend([
                    "Bright top with dark jeans",
                    "Off-shoulder top with A-line skirt",
                    "Structured jacket with flattering neckline"
                ])
            elif body_shape == 'hourglass':
                examples.extend([
                    "Wrap dress with belt",
                    "Fitted dress with defined waist",
                    "Fitted top with pencil skirt"
                ])
        
        return examples[:3]  # Return top 3 simple examples
    
    def _get_gender_specific_recommendations(self, base_recommendations, gender, body_shape, face_shape):
        """Apply gender-specific modifications to recommendations using separate rule sets"""
        rec = base_recommendations.copy()
        
        # Select gender-specific rules
        shape_rules = self.male_shape_rules if gender == 'male' else self.female_shape_rules
        footwear_rules = self.male_footwear_rules if gender == 'male' else self.female_footwear_rules
        
        # Face rules are gender-neutral (same for both)
        face_rules = {
            'Oval': {
                'necklines': ['All necklines work well', 'V-neck', 'Round neck', 'Square neck'],
                'avoid': ['Extreme necklines that disrupt balance'],
                'hair_accessories': ['All styles work', 'Side parts', 'Center parts'],
                'glasses': ['Oval frames', 'Rectangular frames', 'Wayfarer style']
            },
            'Round': {
                'necklines': ['V-neck', 'Deep V-neck', 'Square neck', 'Off-shoulder'],
                'avoid': ['Round necklines', 'High necklines'],
                'hair_accessories': ['Angular hairstyles', 'High ponytails', 'Side-swept bangs'],
                'glasses': ['Angular frames', 'Rectangular frames', 'Cat-eye']
            },
            'Square': {
                'necklines': ['Round neck', 'V-neck', 'Scoop neck', 'Off-shoulder'],
                'avoid': ['Square necklines', 'Straight necklines'],
                'hair_accessories': ['Soft waves', 'Layered hairstyles', 'Side parts'],
                'glasses': ['Round frames', 'Oval frames', 'Aviators']
            },
            'Heart': {
                'necklines': ['V-neck', 'Scoop neck', 'Sweetheart neckline', 'Off-shoulder'],
                'avoid': ['High necklines', 'Turtle necks'],
                'hair_accessories': ['Side parts', 'Wispy bangs', 'Chin-length bobs'],
                'glasses': ['Round bottom frames', 'Aviators', 'Light-colored frames']
            }
        }
        
        # Apply gender-specific body shape guidance
        if body_shape in shape_rules:
            rec['fit_recommendations'] = shape_rules[body_shape]['fit']
            rec['avoid_fits'] = shape_rules[body_shape]['avoid']
            rec['silhouette_guide'] = shape_rules[body_shape]['silhouette']
            rec['accessory_suggestions'] = shape_rules[body_shape]['accessories']
        
        # Apply gender-specific footwear guidance
        if body_shape in footwear_rules:
            rec['footwear_recommendations'] = footwear_rules[body_shape]
        
        # Apply face shape guidance
        if face_shape in face_rules:
            rec['neckline_recommendations'] = face_rules[face_shape]['necklines']
            rec['avoid_necklines'] = face_rules[face_shape]['avoid']
            rec['hair_glasses_tips'] = face_rules[face_shape]
        
        # Mark gender
        rec['gender'] = gender
        
        return rec
    
    def get_recommendation(self, skin_tone, body_shape, face_shape, gender='female'):
        """Get personalized fashion recommendation with gender-specific adjustments"""
        key = f"{skin_tone}_{body_shape}_{face_shape}"
        base_recommendations = self.rules.get(key, {})
        
        if not base_recommendations:
            return {}
        
        # Apply gender-specific modifications with body shape context
        recommendations = self._get_gender_specific_recommendations(base_recommendations, gender, body_shape, face_shape)
        
        # Get gender-specific recommended styles
        recommendations['recommended_styles'] = self._get_recommended_styles(
            skin_tone, body_shape, gender
        )
        
        # Update outfit examples for gender
        recommendations['outfit_examples'] = self._generate_outfit_examples(
            skin_tone, body_shape, face_shape, 
            recommendations.get('recommended_styles', []), 
            gender
        )

        # Add folder path for dynamic image loading
        # Format: skintone_face_shape_body_shape (e.g., light_Oval_apple)
        folder_name = f"{skin_tone}_{face_shape}_{body_shape}"
        recommendations['style_image_folder'] = folder_name
        recommendations['style_image_gender'] = gender

        return recommendations

# ==================== INITIALIZE MODELS ====================

skin_detector = SkinToneDetector(r"graceapp\deit_skin_tone_classifier.pth")
body_detector = BodyShapeDetector(r"graceapp\shape_model.pt")
face_shape_detector = FaceShapeDetector()
rule_engine = FashionRuleEngine()

# ==================== DJANGO VIEWS ====================

def home(request):
    """Render the main page"""
    session_id = request.session.get('session_id', str(uuid.uuid4()))
    request.session['session_id'] = session_id
    
    return render(request, 'home.html', {'session_id': session_id})

@csrf_exempt
def upload_face_image(request):
    """Handle face image upload and skin tone + face shape detection"""
    if request.method == 'POST' and request.FILES.get('face_image'):
        session_id = request.POST.get('session_id')
        
        # Save uploaded image
        uploaded_file = request.FILES['face_image']
        file_name = default_storage.save(f'temp/{uploaded_file.name}', ContentFile(uploaded_file.read()))
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)
        
        # Detect skin tone
        skin_result = skin_detector.predict(file_path)
        
        # Detect face shape
        face_shape_result = detect_face_shape(file_path)
        
        if skin_result and face_shape_result:
            # Save to database
            user_image = UserImage.objects.create(
                session_id=session_id,
                image_type='face',
                image=uploaded_file,
                skin_tone=skin_result['skin_tone'],
                face_shape=face_shape_result['face_shape']
            )
            
            # Clean up temp file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return JsonResponse({
                'success': True,
                'skin_tone': skin_result['skin_tone'],
                'skin_confidence': skin_result['confidence'],
                'face_shape': face_shape_result['face_shape'],
                'face_confidence': face_shape_result['confidence'],
                'message': 'Face image uploaded successfully'
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request'})

def detect_face_shape(image_path):
    """Detect face shape from image"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Detect landmarks
        landmarks = face_shape_detector.detect_landmarks(image)
        if landmarks is None:
            return None
        
        # Calculate metrics
        metrics = face_shape_detector.calculate_face_metrics(landmarks, image.shape)
        
        # Classify face shape
        face_shape, confidence = face_shape_detector.classify_face_shape(metrics)
        
        return {
            'face_shape': face_shape,
            'confidence': confidence,
            'metrics': metrics
        }
    except Exception as e:
        print(f"Error in face shape detection: {e}")
        return None

@csrf_exempt
def upload_body_image(request):
    """Handle body image upload and shape detection"""
    if request.method == 'POST' and request.FILES.get('body_image'):
        session_id = request.POST.get('session_id')
        
        # Save uploaded image
        uploaded_file = request.FILES['body_image']
        file_name = default_storage.save(f'temp/{uploaded_file.name}', ContentFile(uploaded_file.read()))
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)
        
        # Detect body shape
        result = body_detector.predict(file_path)
        
        if result:
            # Save to database
            user_image = UserImage.objects.create(
                session_id=session_id,
                image_type='body',
                image=uploaded_file,
                body_shape=result['body_shape']
            )
            
            # Clean up temp file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return JsonResponse({
                'success': True,
                'body_shape': result['body_shape'],
                'confidence': result['confidence'],
                'message': 'Body image uploaded successfully'
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request'})

def get_recommendations(request):
    """Get personalized fashion recommendations"""
    session_id = request.GET.get('session_id')
    
    if not session_id:
        return JsonResponse({'error': 'Session ID required'}, status=400)
    
    # Get latest results for this session
    face_image = UserImage.objects.filter(
        session_id=session_id, 
        image_type='face'
    ).last()
    
    body_image = UserImage.objects.filter(
        session_id=session_id, 
        image_type='body'
    ).last()
    
    if not face_image or not body_image:
        return JsonResponse({
            'error': 'Please upload both face and body images',
            'has_face': bool(face_image),
            'has_body': bool(body_image)
        }, status=400)
    
    # Get gender from session (default to female if not set)
    gender = request.session.get(f'gender_{session_id}', 'female')
    
    # Get recommendations with gender
    recommendations = rule_engine.get_recommendation(
        face_image.skin_tone,
        body_image.body_shape,
        face_image.face_shape,
        gender
    )
    
    # Save recommendation to database
    if recommendations:
        FashionRecommendation.objects.create(
            skin_tone=face_image.skin_tone,
            body_shape=body_image.body_shape,
            face_shape=face_image.face_shape,
            recommendation=json.dumps(recommendations)
        )
    
    return JsonResponse({
        'success': True,
        'skin_tone': face_image.skin_tone,
        'body_shape': body_image.body_shape,
        'face_shape': face_image.face_shape,
        'recommendations': recommendations
    })

def clear_session(request):
    """Clear session data"""
    if 'session_id' in request.session:
        del request.session['session_id']
    return JsonResponse({'success': True})

@csrf_exempt
def set_gender(request):
    """Store user's gender selection in session"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session_id = data.get('session_id')
            gender = data.get('gender')
            
            if session_id and gender in ['male', 'female']:
                # Store gender in session
                request.session[f'gender_{session_id}'] = gender
                return JsonResponse({'success': True, 'gender': gender})
            else:
                return JsonResponse({'success': False, 'error': 'Invalid session or gender'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

def get_style_image(request):
    """Serve a random style image from the appropriate folder based on detection results"""
    folder = request.GET.get('folder', '')  # e.g., "light_Oval_apple"
    gender = request.GET.get('gender', 'female')  # "male" or "female"

    if not folder:
        return JsonResponse({'success': False, 'error': 'Folder parameter required'}, status=400)

    # Base directory for style images (you need to create this structure)
    # Structure: media/style_images/{folder}/{gender}/image.jpg
    base_dir = os.path.join(settings.MEDIA_ROOT, 'style_images')
    target_dir = os.path.join(base_dir, folder, gender)

    # Print folder path to terminal for debugging
    print(f"[Style Image] Looking for images in: {target_dir}")
    print(f"[Style Image] Absolute path: {os.path.abspath(target_dir)}")

    # Auto-create folder structure if it doesn't exist
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir, exist_ok=True)
            print(f"[Style Image] Created folder: {target_dir}")
        except Exception as e:
            print(f"[Style Image] Error creating folder: {e}")

    # Check if directory exists and has images
    if os.path.exists(target_dir):
        # Get all image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
        try:
            images = [f for f in os.listdir(target_dir) if f.lower().endswith(valid_extensions)]
            print(f"[Style Image] Found {len(images)} images: {images}")
        except Exception as e:
            print(f"[Style Image] Error listing directory: {e}")
            images = []

        if images:
            # Pick random image
            random_image = random.choice(images)
            image_path = os.path.join(target_dir, random_image)
            print(f"[Style Image] Selected image: {random_image}")

            # Return the image file
            try:
                with open(image_path, 'rb') as f:
                    content_type = 'image/jpeg'
                    if random_image.lower().endswith('.png'):
                        content_type = 'image/png'
                    elif random_image.lower().endswith('.gif'):
                        content_type = 'image/gif'
                    elif random_image.lower().endswith('.webp'):
                        content_type = 'image/webp'

                    print(f"[Style Image] Returning image with content-type: {content_type}")
                    return HttpResponse(f.read(), content_type=content_type)
            except Exception as e:
                print(f"[Style Image] Error reading image file: {e}")
                return JsonResponse({'success': False, 'error': f'Error reading image: {str(e)}'}, status=500)

    # If no custom images found, return placeholder response with folder info
    return JsonResponse({
        'success': False,
        'error': 'No images found in folder',
        'folder': folder,
        'gender': gender,
        'folder_created': True,
        'expected_path': f'/media/style_images/{folder}/{gender}/',
        'absolute_path': os.path.abspath(target_dir),
        'message': 'Folder created. Please upload images to this folder to see style recommendations.'
    }, status=404)


@csrf_exempt
def upload_style_image(request):
    """Upload an image to a specific style folder"""
    if request.method == 'POST' and request.FILES.get('style_image'):
        folder = request.POST.get('folder', '')
        gender = request.POST.get('gender', 'female')

        if not folder:
            return JsonResponse({'success': False, 'error': 'Folder parameter required'}, status=400)

        # Create target directory
        base_dir = os.path.join(settings.MEDIA_ROOT, 'style_images')
        target_dir = os.path.join(base_dir, folder, gender)

        # Create folders if don't exist
        os.makedirs(target_dir, exist_ok=True)

        # Save uploaded image
        uploaded_file = request.FILES['style_image']
        file_path = os.path.join(target_dir, uploaded_file.name)

        try:
            with open(file_path, 'wb+') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            print(f"[Style Upload] Image saved to: {file_path}")

            return JsonResponse({
                'success': True,
                'message': f'Image uploaded to {folder}/{gender}/',
                'filename': uploaded_file.name,
                'folder': folder,
                'gender': gender,
                'full_path': file_path
            })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)

    return JsonResponse({'success': False, 'error': 'Invalid request'}, status=400)