from typing import Tuple
from django.db import models
from django.db.models import fields
from departments.models import Department
class Camera(models.Model):
    ip = models.GenericIPAddressField(null=False)
    username = models.CharField(max_length=50)
    pwd = models.CharField(max_length=50)
    dept = models.ForeignKey(Department, on_delete=models.CASCADE)
    name = models.CharField(max_length=200, default='Anonymous')
    location = models.CharField(max_length=300, default="undefined")
    def __str__(self):
        return self.ip

class OfflineCameras(models.Model):
    camera = models.ForeignKey(Camera, on_delete= models.CASCADE)
    time = models.DateTimeField(null=True)
    sms_sent = models.BooleanField(default=False)

    def __str__(self):
        return str(self.camera)

class OnlineCameras(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    
    def __str__(self):
        return str(self.camera)
