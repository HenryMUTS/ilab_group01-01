from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import MeViewSet, PatientProfileViewSet, ConsentViewSet, PhotoViewSet, PredictionViewSet, SignupView


router = DefaultRouter()
router.register(r'me', MeViewSet, basename='me')
router.register(r'profile', PatientProfileViewSet, basename='profile')
router.register(r'consents', ConsentViewSet, basename='consents')
router.register(r'photos', PhotoViewSet, basename='photos')
router.register(r'predictions', PredictionViewSet, basename='predictions')


urlpatterns = [
path('auth/signup/', SignupView.as_view(), name='signup'),
path('', include(router.urls)),
]