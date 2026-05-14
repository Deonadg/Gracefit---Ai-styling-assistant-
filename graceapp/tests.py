from django.test import TestCase

# Create your tests here.
# test_skin_tone.py
import sys
import os

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graceapp.views import SkinToneDetector

# Test with your model
detector = SkinToneDetector(r"graceapp\best_model.pth")

# Test with a sample image
test_image = r"graceapp\brownimage.jpg.chip.jpg"  # or any test image
result = detector.predict(test_image)

if result:
    print("\n" + "="*50)
    print("FINAL RESULT")
    print("="*50)
    print(f"Predicted Skin Tone: {result['skin_tone']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Available Classes: {result['all_classes']}")
    print("="*50)
    
    # Print all probabilities
    print("\nAll Class Probabilities:")
    for i, (class_name, prob) in enumerate(zip(result['all_classes'], result['all_probs'])):
        print(f"{i}. {class_name}: {prob:.2%}")