from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    ROLE_CHOICES = (("customer","Customer"), ("admin","Doctor/Admin"))
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default="customer")


class PatientProfile(models.Model):
    GENDER_CHOICES = (
        ("female", "Female"),
        ("male", "Male"),
        ("unspecified", "Unspecified"),
    )
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    dob = models.DateField()
    phone = models.CharField(max_length=32, blank=True)
    medical_conditions = models.TextField(blank=True)
    drug_allergies = models.TextField(blank=True)
    under18 = models.BooleanField(default=False)
    parent_name = models.CharField(max_length=128, blank=True)
    parent_consented = models.BooleanField(default=False)


class ConsentReceipt(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='consents')
    version = models.CharField(max_length=16, default='v1')
    agreed_at = models.DateTimeField(auto_now_add=True)
    ip_addr = models.GenericIPAddressField(null=True, blank=True)


class Photo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='photos')
    original = models.ImageField(upload_to='uploads/')
    created_at = models.DateTimeField(auto_now_add=True)


class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    photo = models.ForeignKey(Photo, on_delete=models.CASCADE, related_name='predictions')
    output = models.ImageField(upload_to='outputs/')
    metrics = models.JSONField(default=dict)
    model_version = models.CharField(max_length=32, default='demo-0.1')
    created_at = models.DateTimeField(auto_now_add=True)