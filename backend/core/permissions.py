from rest_framework.permissions import BasePermission, SAFE_METHODS


class IsAdminOrOwner(BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.user and request.user.is_authenticated and request.user.role == 'admin':
            return True
        return getattr(obj, 'user_id', None) == request.user.id


    def has_permission(self, request, view):
        return request.user and request.user.is_authenticated