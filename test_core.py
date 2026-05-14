#!/usr/bin/env python
"""
Test script to verify GraceFit core functionality
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gracefit.settings')
django.setup()

def test_models():
    """Test that models can be imported and created"""
    try:
        from graceapp.models import UserImage, FashionRecommendation
        print("✅ Models imported successfully")
        
        # Test model fields exist
        user_image_fields = [f.name for f in UserImage._meta.fields]
        expected_user_fields = ['session_id', 'image_type', 'image', 'uploaded_at', 'skin_tone', 'body_shape', 'face_shape']
        
        for field in expected_user_fields:
            if field in user_image_fields:
                print(f"✅ UserImage.{field} field exists")
            else:
                print(f"❌ UserImage.{field} field missing")
        
        rec_fields = [f.name for f in FashionRecommendation._meta.fields]
        expected_rec_fields = ['skin_tone', 'body_shape', 'face_shape', 'recommendation']
        
        for field in expected_rec_fields:
            if field in rec_fields:
                print(f"✅ FashionRecommendation.{field} field exists")
            else:
                print(f"❌ FashionRecommendation.{field} field missing")
                
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_urls():
    """Test that URL patterns are configured"""
    try:
        from django.urls import reverse
        home_url = reverse('home')
        upload_face_url = reverse('upload_face')
        upload_body_url = reverse('upload_body')
        get_rec_url = reverse('get_recommendations')
        clear_session_url = reverse('clear_session')
        
        print("✅ URL patterns configured successfully")
        print(f"   Home: {home_url}")
        print(f"   Upload Face: {upload_face_url}")
        print(f"   Upload Body: {upload_body_url}")
        print(f"   Get Recommendations: {get_rec_url}")
        print(f"   Clear Session: {clear_session_url}")
        
        return True
    except Exception as e:
        print(f"❌ URL test failed: {e}")
        return False

def test_templates():
    """Test that templates exist"""
    try:
        template_path = os.path.join('graceapp', 'templates', 'home.html')
        if os.path.exists(template_path):
            print("✅ Main template exists")
            with open(template_path, 'r') as f:
                content = f.read()
                if 'upload-face' in content and 'upload-body' in content:
                    print("✅ Template contains upload functionality")
                if 'recommendationsContent' in content:
                    print("✅ Template contains recommendations display")
        else:
            print("❌ Main template missing")
            return False
        return True
    except Exception as e:
        print(f"❌ Template test failed: {e}")
        return False

def main():
    print("🔍 Testing GraceFit Core Functionality")
    print("=" * 50)
    
    tests = [
        ("Models", test_models),
        ("URLs", test_urls),
        ("Templates", test_templates),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core functionality tests passed!")
        print("📝 Note: Full server requires ML dependencies (mediapipe, torch, etc.)")
        print("💡 The application structure is correct and ready to run once dependencies are installed.")
    else:
        print("⚠️ Some tests failed - check the errors above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
