from io import BytesIO
from PIL import Image, ImageOps
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from django.core.files.base import ContentFile
from django.contrib.auth import get_user_model
from rest_framework import viewsets, mixins, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import PatientProfile, ConsentReceipt, Photo, Prediction
from .serializers import (
    UserSerializer, PatientProfileSerializer, ConsentReceiptSerializer,
    PhotoSerializer, PredictionSerializer, SignupSerializer
)
from .permissions import IsAdminOrOwner

class SignupView(APIView):
    permission_classes = [AllowAny]


    def post(self, request):
        ser = SignupSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        user = ser.save()
        return Response(UserSerializer(user).data, status=201)

User = get_user_model()

class MeViewSet(viewsets.ViewSet):
    permission_classes = [permissions.IsAuthenticated]


    def list(self, request):
        return Response(UserSerializer(request.user).data)


class PatientProfileViewSet(mixins.CreateModelMixin,
                            mixins.RetrieveModelMixin,
                            mixins.UpdateModelMixin,
                            viewsets.GenericViewSet):
    serializer_class = PatientProfileSerializer
    permission_classes = [permissions.IsAuthenticated]


    def get_object(self):
        obj, _ = PatientProfile.objects.get_or_create(user=self.request.user)
        return obj
    
class ConsentViewSet(mixins.CreateModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    serializer_class = ConsentReceiptSerializer
    permission_classes = [permissions.IsAuthenticated]


    def get_queryset(self):
        qs = ConsentReceipt.objects.all()
        if self.request.user.role != 'admin':
            qs = qs.filter(user=self.request.user)
            return qs.order_by('-agreed_at')


    def perform_create(self, serializer):
        ip = self.request.META.get('REMOTE_ADDR')
        serializer.save(user=self.request.user, ip_addr=ip)


class PhotoViewSet(mixins.CreateModelMixin,
                    mixins.RetrieveModelMixin,
                    mixins.ListModelMixin,
                    viewsets.GenericViewSet):
    serializer_class = PhotoSerializer
    permission_classes = [permissions.IsAuthenticated]


    def get_queryset(self):
        qs = Photo.objects.all()
        if self.request.user.role != 'admin':
            qs = qs.filter(user=self.request.user)
        return qs.order_by('-created_at')


    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class PredictionViewSet(mixins.CreateModelMixin,
                        mixins.RetrieveModelMixin,
                        mixins.ListModelMixin,
                        viewsets.GenericViewSet):
    serializer_class = PredictionSerializer
    permission_classes = [permissions.IsAuthenticated]


    def get_queryset(self):
        qs = Prediction.objects.select_related('photo','user').all()
        if self.request.user.role != 'admin':
            qs = qs.filter(user=self.request.user)
        return qs.order_by('-created_at')

    def create(self, request, *args, **kwargs):
    # Expect {"photo_id": <id>}
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        photo = serializer.validated_data['photo']


        if request.user.role != 'admin' and photo.user_id != request.user.id:
            return Response({"detail": "Not your photo."}, status=status.HTTP_403_FORBIDDEN)


        # Load original image
        img = Image.open(photo.original).convert('RGB')
        # === Demo inference: equalize + light blur (replace with real model) ===
        out = ImageOps.equalize(img)
        out = out.filter(Image.Filter.BoxBlur(1))
        # Save to memory
        buf = BytesIO()
        out.save(buf, format='PNG')
        buf.seek(0)


        # Metrics stub
        metrics = {"psnr": 30.5, "lpips": 0.22, "inference_ms": 95, "model_version": "demo-0.1"}


        pred = Prediction(user=request.user, photo=photo, metrics=metrics, model_version=metrics['model_version'])
        pred.output.save(f"pred_{photo.id}.png", ContentFile(buf.read()), save=True)
        pred.save()


        return Response(PredictionSerializer(pred).data, status=status.HTTP_201_CREATED)