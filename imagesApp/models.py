from django.db import models

from cameras.models import Camera


# Create your models here.
class Person(models.Model):
    person_id = models.AutoField(primary_key=True)
    date_time = models.DateTimeField()
    person_img = models.ImageField(upload_to="personImages/", default="")
    face_img = models.ImageField(upload_to="personImages/", null=True, blank=True)
    conf_score = models.FloatField()
    camera_id = models.ForeignKey(Camera, on_delete=models.CASCADE, null=True, blank=True)
    
    def __str__(self):
        return str(self.person_id)