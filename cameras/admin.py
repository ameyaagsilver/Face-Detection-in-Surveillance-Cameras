from django.contrib import admin

from cameras.models import Camera, OfflineCameras, OnlineCameras

# Register your models here.

admin.site.register(OfflineCameras)
admin.site.register(OnlineCameras)
admin.site.register(Camera)