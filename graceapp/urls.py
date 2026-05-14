from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload-face/', views.upload_face_image, name='upload_face'),
    path('upload-body/', views.upload_body_image, name='upload_body'),
    path('get-recommendations/', views.get_recommendations, name='get_recommendations'),
    path('clear-session/', views.clear_session, name='clear_session'),
    path('set-gender/', views.set_gender, name='set_gender'),
    path('get-style-image/', views.get_style_image, name='get_style_image'),
    path('upload-style-image/', views.upload_style_image, name='upload_style_image'),
]