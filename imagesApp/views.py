import json
from django.shortcuts import render
from django.core.paginator import Paginator
from django.utils import timezone
from jmespath import search

from urllib3 import HTTPResponse

from cctv.settings import MEDIA_ROOT
from .models import Person
from django.core import serializers
# limit the number of cpus used by high performance libraries
from datetime import datetime
import os
os.environ["OMP_NUM_THREADS"] = "9"
os.environ["OPENBLAS_NUM_THREADS"] = "9"
os.environ["MKL_NUM_THREADS"] = "9"
os.environ["VECLIB_MAXIMUM_THREADS"] = "9"
os.environ["NUMEXPR_NUM_THREADS"] = "9"
# from .views import cameraInUse
import sys
sys.path.insert(0, './yolov5')
import argparse
import os
import platform
import shutil
import time
from pathlib import Path, WindowsPath
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# *********************************
from django.db.models import Max

# Create your views here.
# output='inference/output'
# source='rtsp://username:password@' + str(ip) + ':554/uservid=False,=admin_password=@dmin123_channel=channel_number_stream=1.sdp'
path = []
agnostic_nms=False
augment=False
# path.append(source)
# yolo_model='yolov5l.pt'
# yolo_model = "C:/Users/WS-AIML-ISE/Desktop/Minor Project - CCTV Research Project/cctv_django-main/cameras/yolov5m-face.pt"
# deep_sort_model='osnet_x0_25'
device=''
show_vid=False
iou_thres=0.3
conf_thres=0.15
imgsz=[640, 640]
max_det=1000
evaluate=False
half=False
project=WindowsPath('runs/track')
visualize=False
exist_ok=False
update=False
save_txt=False
save_crop=False
dnn=False
# config_deepsort='C:/Users/WS-AIML-ISE/Desktop/Minor Project - CCTV Research Project/cctv_django-main/cameras/deep_sort.yaml'
classes = None
device = select_device(device)
half &= device.type != 'cpu'


yolo_face_model_name = str(ROOT) + '/yolov5m-face.pt'
print(yolo_face_model_name)
face_data = "C:/Users/WS-AIML-ISE/Desktop/Minor Project - CCTV Research Project/cctv_django-main/faceDataset.yaml"
yolo_face_model = DetectMultiBackend(yolo_face_model_name, device=device, dnn=dnn, data=face_data, fp16=half)
f_stride, f_names, f_pt = yolo_face_model.stride, yolo_face_model.names, yolo_face_model.pt
imgsz = check_img_size(imgsz, s=f_stride)  # check image size
# print(yolo_face_model)

def getAllPersons(request):
    all_persons = Person.objects.all()
    context = {}
    searchedQuery = {}
    persons_list1 = Person.objects.none()
    persons_list2 = Person.objects.none()
    persons_list3 = Person.objects.none()
    persons_list4 = Person.objects.none()

    if request.GET.get('startDateTime'):
        searchedQuery['startDateTime'] = request.GET.get('startDateTime')
        t_index = request.GET.get('startDateTime').index('T')
        date = request.GET.get('startDateTime')[0:t_index]
        time = request.GET.get('startDateTime')[t_index+1:]
        date_time_obj = datetime.strptime(date[2:]+" "+time, '%y-%m-%d %H:%M')
        # print(type(date_time_obj))
        # print(date_time_obj)
        today_max = datetime.combine(timezone.now().date(), datetime.today().time().max)
        persons_list1 = persons_list1 | Person.objects.filter(date_time__range=(date_time_obj, today_max))
        # print("List 1 :: ", persons_list1)
    else:
        persons_list1 = Person.objects.all()

    if request.GET.get('endDateTime'):
        searchedQuery['endDateTime'] = request.GET.get('endDateTime')
        t_index = request.GET.get('endDateTime').index('T')
        date = request.GET.get('endDateTime')[0:t_index]
        time = request.GET.get('endDateTime')[t_index+1:]
        date_time_obj = datetime.strptime(date[2:]+" "+time, '%y-%m-%d %H:%M')
        # print(type(date_time_obj))
        # print(date_time_obj)
        min_date = datetime.strptime("00-01-01 00:00", '%y-%m-%d %H:%M')
        persons_list2 = persons_list2 | Person.objects.filter(date_time__range=(min_date, date_time_obj))
        # print("List 2 :: ", persons_list2)
    else:
        persons_list2 = Person.objects.all()

    if request.GET.get('personID'):
        searchedQuery['personID'] = request.GET.get('personID')
        # print(request.GET.get('personID'))
        personID = request.GET.get('personID')
        persons_list3 = persons_list3 | Person.objects.filter(person_id=personID)
    else:
        persons_list3 = Person.objects.all()

    if request.GET.get('confScore'):
        searchedQuery['confScore'] = request.GET.get('confScore')
        confScore = float(request.GET.get('confScore'))/100
        # print(confScore)
        # print(type(confScore))
        persons_list4 = persons_list4 | Person.objects.filter(conf_score__gte=confScore)
    else:
        persons_list4 = Person.objects.all()
    print("searchedQuery", searchedQuery)
    all_persons = all_persons & persons_list1 & persons_list2 & persons_list3 & persons_list4
    paginator = Paginator(all_persons, 15)
    page = request.GET.get('page')
    paginated_all_persons = paginator.get_page(page)

    # return render(request, 'get-all-persons.html', context)
    context['persons'] = paginated_all_persons
    context['searchedQuery'] = searchedQuery
    return render(request, 'getAllPersonsSenEdition.html', context)
#'getAllPersonsSenEdition.html'

def getMoreInfoOnPerson(request, personId):
    person = Person.objects.get(person_id=personId)
    print(type(person.date_time))
    print(person.date_time)
    context = {}
    context['person'] = person
    if person.face_img is not None and len(str(person.face_img))>0:
        return render(request, 'detailedDescriptionSenEdition.html', {"person" : person})
    face_source = MEDIA_ROOT + "\\" + str(person.person_img)
    print(face_source, "FACE SOURCEEE")
    face_dataset = LoadImages(face_source, img_size=imgsz, stride=f_stride, auto=f_pt)
    print(len(face_dataset), "Face DATASET Length")
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in face_dataset:
        cropped_person_img = im
        # print(cropped_person_img)
        cropped_person_img = torch.from_numpy(cropped_person_img).to(device)
        cropped_person_img = cropped_person_img.half() if yolo_face_model.fp16 else cropped_person_img.float()  # uint8 to fp16/32
        cropped_person_img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(cropped_person_img.shape) == 3:
            cropped_person_img = cropped_person_img[None]  # expand for batch dim
        face_pred = yolo_face_model(cropped_person_img, augment=augment, visualize=visualize)

        face_pred = non_max_suppression(face_pred, conf_thres, iou_thres, None, agnostic_nms, max_det=max_det)
        print(face_pred, "FACE PREDICTIONS")
        print("Done till here!!!!!!!")
        for i, det in enumerate(face_pred):  # per image
            seen += 1
            
            p, im0, frame = path, im0s.copy(), getattr(face_dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if face_dataset.mode == 'image' else f'_{frame}')  # im.txt
            # s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=2, example=str(f_names))
            # print(det)
            if len(det):
                # Rescale boxes from img_size to im0 size
                # print(cropped_person_img.shape[0:], det[:, :4], im0.shape)
                # print(cropped_person_img)
                det[:, :4] = scale_coords(cropped_person_img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {f_names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy[0] = torch.tensor(xyxy[0].item()-5)
                    xyxy[1] = torch.tensor(xyxy[1].item()-5)
                    xyxy[2] = torch.tensor(xyxy[2].item()+5)
                    xyxy[3] = torch.tensor(xyxy[3].item()+5)
                    # print(xyxy)
                    if save_crop:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if True else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if True:
                        face_image_location = str('crops' + "/" + 'person' + "/" + str(person.person_id) + '.0' + "/" + f'face_crop.jpg')
                        save_one_box(xyxy, imc, file=WindowsPath(MEDIA_ROOT) / 'crops' / 'person' / (str(person.person_id) + '.0') / f'face_crop.jpg', BGR=True)
                        person.face_img = face_image_location
                        person.save()

    return render(request, 'detailedDescriptionSenEdition.html', context)