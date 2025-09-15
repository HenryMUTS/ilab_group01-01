from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from .models import PatientProfile, ConsentReceipt, Photo, Prediction


User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ("id","username","email","role")


class PatientProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = PatientProfile
        fields = "__all__"
        read_only_fields = ("user","under18")

class SignupSerializer(serializers.Serializer):
    username = serializers.CharField()
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)
    dob = serializers.DateField()
    phone = serializers.CharField(required=False, allow_blank=True)
    gender = serializers.ChoiceField(choices=PatientProfile.GENDER_CHOICES, default="unspecified")
    medical_conditions = serializers.CharField(required=False, allow_blank=True)
    drug_allergies = serializers.CharField(required=False, allow_blank=True)


    def validate_password(self, value):
        validate_password(value)
        return value


    def create(self, validated_data):
        username = validated_data["username"]
        email = validated_data["email"]
        password = validated_data["password"]
        dob = validated_data["dob"]
        phone = validated_data.get("phone", "")
        gender = validated_data.get("gender", "unspecified")
        medical_conditions = validated_data.get("medical_conditions", "")
        drug_allergies = validated_data.get("drug_allergies", "")


        user = User.objects.create_user(username=username, email=email, password=password)
        # default role is customer
        user.role = getattr(user, 'role', 'customer') or 'customer'
        user.save()


        # compute under18
        from datetime import date
        today = date.today()
        under18 = (today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))) < 18


        PatientProfile.objects.create(
            user=user,
            dob=dob,
            phone=phone,
            gender=gender,
            medical_conditions=medical_conditions,
            drug_allergies=drug_allergies,
            under18=under18,
        )
        return user

class ConsentReceiptSerializer(serializers.ModelSerializer):
    class Meta:
        model = ConsentReceipt
        fields = "__all__"
        read_only_fields = ("user","agreed_at","ip_addr")


class PhotoSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    class Meta:
        model = Photo
        fields = ("id","user","original","created_at")
        read_only_fields = ("id","user","created_at")


class PredictionSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    photo = PhotoSerializer(read_only=True)
    photo_id = serializers.PrimaryKeyRelatedField(queryset=Photo.objects.all(), write_only=True, source='photo')


    class Meta:
        model = Prediction
        fields = ("id","user","photo","photo_id","output","metrics","model_version","created_at")
    read_only_fields = ("id","user","photo","output","metrics","model_version","created_at")