from django.contrib import admin
from .models import User, PatientProfile, ConsentReceipt, Photo, Prediction


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ("username","email","role","is_active")


@admin.register(PatientProfile)
class PatientProfileAdmin(admin.ModelAdmin):
    list_display = ("user","dob","under18","parent_consented")


@admin.register(ConsentReceipt)
class ConsentAdmin(admin.ModelAdmin):
    list_display = ("user","version","agreed_at","ip_addr")


@admin.register(Photo)
class PhotoAdmin(admin.ModelAdmin):
    list_display = ("id","user","created_at","original")


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("id","user","photo","created_at","model_version")