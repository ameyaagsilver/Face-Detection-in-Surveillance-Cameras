from email import contentmanager
from os import name
from django.contrib.messages.api import error
from django.http.response import Http404, HttpResponse
from django.views.decorators import gzip
from django.shortcuts import redirect, render
from django.views import View
from django.urls import reverse
from django.http import StreamingHttpResponse
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.contrib.auth.decorators import login_required

from cameras.forms import CameraDetailsEntry, CameraDetailsEdit
from departments.models import Department
from .models import Camera, OfflineCameras, OnlineCameras
from .videoCamera import VideoCamera

import platform
import pandas as pd
import subprocess
from threading import Thread
import datetime
from .sms import broadcast_sms
# Create your views here.

# cameraInUse = []


class AddNewCamera(View, LoginRequiredMixin):
    def get(self, request):
        context = dict()
        deptList = Department.objects.all()
        context['deptList'] = deptList
        form = CameraDetailsEntry()
        context['form'] = form
        return render(request, 'cameras/newCamera.html', context)

    def post(self, request):
        form = CameraDetailsEntry(request.POST)
        if form.is_valid():
            form.save()
        return redirect(reverse('new-camera'))

class StatusView(View, LoginRequiredMixin):
    onlineCameraList = []
    offlineCameraList = []
    sms_tbs_CamerasList = []
    threads = []
    def ping(self, camera):

        param = '-n' if platform.system().lower() == 'windows' else 'c'
        command = ['ping', param, '1', camera.ip]

        if subprocess.call(command) == 0:

            #===============        Log camera as online    ================================
            file1 = open(r"logs/generalLog.txt", "a")
            file1.write("["+ str(datetime.datetime.now()) + "]: "+ "Camera "+ str(camera.id)+ "; IP Address: "+ str(camera.ip)+ " Online\n")
            file1.close()

            #===============    Add camera to Online Cameras table =======================
            OnlineCameras.objects.get_or_create(camera = camera)

            #===============    Delete Camera from Offline cameras =======================
            if OfflineCameras.objects.filter(camera = camera).exists():
                #log that camera came online
                OfflineCameras.objects.get(camera = camera).delete()

            #append camera to context variable
            self.onlineCameraList.append(camera)

        else:
            #===============        Log camera as offline    ================================
                file1 = open(r"logs/generalLog.txt", "a")
                file1.write("["+ str(datetime.datetime.now()) + "]: "+ "Camera "+ str(camera.id)+ "; IP Address: "+ str(camera.ip)+ " Offline\n")
                file1.close()

                #===============    Add camera to Offline Cameras table =======================
                OfflineCameras.objects.get_or_create(camera = camera)
                tempCam = OfflineCameras.objects.get(camera = camera)
                tempCam.time = datetime.datetime.now()
                tempCam.save()

                cam = OfflineCameras.objects.get(camera=camera)
                if cam.sms_sent == False:
                    self.sms_tbs_CamerasList.append(camera)

                #===============    Delete Camera from Online cameras =======================
                if OnlineCameras.objects.filter(camera = camera).exists():
                    #send sms
                    #log camera as gone offline
                    OnlineCameras.objects.get(camera = camera).delete()
                
                #append camera to context variable
                self.offlineCameraList.append(camera)

    def get(self, request, department):
        self.onlineCameraList = []
        self.offlineCameraList = []
        deptList = Department.objects.all()
        self.sms_tbs_CamerasList=[]
        context = dict()
        context['deptList'] = deptList
        context['dept'] = department    
        if Department.objects.filter(name=department).exists():
            department = Department.objects.get(name = department)
        else:
            raise Http404('Such a department does not exist')
        for camera in Camera.objects.filter(dept = department):
            p = Thread(target=self.ping, args=(camera, ))
            self.threads.append(p)
            p.start()

        for thread in self.threads:
            thread.join()
        
        if len(self.sms_tbs_CamerasList)>0:
            print("sms sent",self.sms_tbs_CamerasList)
            broadcast_sms(department.name,self.sms_tbs_CamerasList)

        self.sms_tbs_CamerasList=[]
        context['offlineCameras'] = self.offlineCameraList
        context['onlineCameras'] = self.onlineCameraList
        return render(request, 'cameras/cameraStatusView.html', context)

def gen(camera):
    while True:
        frame = camera.get_frame()
        # print("Ïm generating!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def livefeed(request, cameraId):
    try:
        cam = VideoCamera(cameraId)
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        print(e)
        messages.error(request, 'Not possible to stream at this time')
        return redirect(reverse('status-view', args=('ISE')))

class CameraFeed(View, LoginRequiredMixin):
    def get(self, request, cameraId):
        context = dict()
        deptList = Department.objects.all()
        context['deptList'] = deptList
        context['cameraId'] = cameraId
        camera = Camera.objects.get(id=cameraId)
        context['camera'] = camera
        return render(request, 'cameras/cameraFeed.html', context)

    def post(self, request, cameraId):
        camera = Camera.objects.get(id=cameraId)
        dept = str(camera.dept)
        camera.delete()
        return redirect(reverse('status-view', args=[dept]))

class EditCameraDetails(View, LoginRequiredMixin):
    def get(self, request, cameraId):
        context = dict()
        deptList = Department.objects.all()
        context['deptList'] = deptList
        camera = Camera.objects.get(id=cameraId)
        form = CameraDetailsEdit(instance=camera)
        context['camera'] = camera
        context['form'] = form
        return render(request, 'cameras/editCamera.html', context)

    def post(self, request, cameraId):
        camera = Camera.objects.get(id=cameraId)
        dept = str(camera.dept)
        if request.POST.get('delete'):
            camera.delete()
        else:
            form = CameraDetailsEdit(request.POST, instance=camera)
            form.save()
        return redirect(reverse('status-view', args=[dept]))

def addCameraList(request):
    camera_df = pd.read_excel(io='RVCE cctv camera details.xlsx', usecols=['IP'], sheet_name=1)
    dept_df = pd.read_excel(io='RVCE cctv camera details.xlsx', usecols=['Dept_id'], sheet_name=1)
    ip_list = pd.Series(camera_df['IP']).to_list()
    dept_list = pd.Series(dept_df['Dept_id']).to_list()

    for i in range(len(ip_list)):
        department = Department.objects.get(id=dept_list[i])
        Camera.objects.create(ip=ip_list[i], dept=department, username='admin', pwd='@dmin123')
    print('done')
