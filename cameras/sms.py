from django.conf import settings                                                                                                                                                       
from django.http import HttpResponse
# from twilio.rest import Client
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK,HTTP_400_BAD_REQUEST
from .models import Camera, OfflineCameras
import datetime;
  

def broadcast_sms(department,sms_tbsCameraList):
    message1 = "Number of cameras off the network in "+department+" department : "+str(len(sms_tbsCameraList))+"\n [[  "+str(datetime.datetime.now())+"  ]]\n"
    message = "\nList of cameras off the network in "+department+" department : \n\n"
    for cam in sms_tbsCameraList:
        tempCam = OfflineCameras.objects.get(camera = cam)
        message+=str(cam.name)+"  :  "+str(cam.location)+"   -  \n[[  "+str(tempCam.time)+" ]] \n\n"
    message_to_broadcast = (message1+message)
    
    print(message_to_broadcast)

    # client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    
    """ try:
        client.messages.create(to=settings.SMS_BROADCAST_TO_NUMBER,
                                   from_=settings.TWILIO_NUMBER,
                                   body=message_to_broadcast)   

        for camera in sms_tbsCameraList:
            cam = OfflineCameras.objects.get(camera=camera)
            if cam.sms_sent ==  False:
                cam.sms_sent = True
                cam.save()
        return Response("message sent!", HTTP_200_OK)

    except:
        return Response("message failed!", HTTP_400_BAD_REQUEST)  """    