from django.db import models
import uuid
import os

def user_image_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join(f"user_{instance.session_id}", instance.image_type, filename)

class UserImage(models.Model):
    IMAGE_TYPES = [
        ('face', 'Face Image'),
        ('body', 'Body Image'),
    ]
    
    session_id = models.CharField(max_length=100)
    image_type = models.CharField(max_length=10, choices=IMAGE_TYPES)
    image = models.ImageField(upload_to=user_image_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    skin_tone = models.CharField(max_length=20, blank=True, null=True)
    body_shape = models.CharField(max_length=30, blank=True, null=True)
    face_shape = models.CharField(max_length=20, blank=True, null=True)  # NEW FIELD
    
    def __str__(self):
        return f"{self.session_id} - {self.image_type}"

class FashionRecommendation(models.Model):
    skin_tone = models.CharField(max_length=20)
    body_shape = models.CharField(max_length=30)
    face_shape = models.CharField(max_length=20)  # NEW FIELD
    recommendation = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.skin_tone} - {self.body_shape} - {self.face_shape}"