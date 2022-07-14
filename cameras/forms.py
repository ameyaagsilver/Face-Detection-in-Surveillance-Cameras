from django.forms import ModelForm, Form, fields
from .models import Camera

class CameraDetailsEntry(ModelForm):
    class Meta:
        model = Camera
        fields = '__all__'

    def save(self, commit=True):
        camera = super().save(commit=False)
        if commit:
            camera.save()
        return camera

class CameraDetailsEdit(ModelForm):
    class Meta:
        model = Camera
        fields = ['username', 'pwd', 'dept', 'name', 'ip']

