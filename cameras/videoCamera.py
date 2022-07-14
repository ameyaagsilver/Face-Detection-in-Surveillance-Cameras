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
import time
import cv2 as cv
from threading import Thread
from .models import Camera
from imagesApp.models import Person
from django.db.models import Max
from PIL import Image
from cctv.settings import BASE_DIR, MEDIA_ROOT

output='inference/output'
# source='rtsp://username:password@' + str(ip) + ':554/uservid=False,=admin_password=@dmin123_channel=channel_number_stream=1.sdp'
# path = []
agnostic_nms=False
augment=False
# path.append(source)
yolo_model='yolov5l.pt'
deep_sort_model='osnet_x0_25'
device=''
show_vid=False
iou_thres=0.5
conf_thres=0.25
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
config_deepsort='C:/Users/WS-AIML-ISE/Desktop/Minor Project - CCTV Research Project/cctv_django-main/cameras/deep_sort.yaml'
classes = None
device = select_device(device)
half &= device.type != 'cpu'
allPersons = Person.objects.all()
max_person_id = allPersons.aggregate(Max('person_id'))
max_person_id = max_person_id['person_id__max']
print(max_person_id, "Max Person ID")

if type(yolo_model) is str:  # single yolo model
    exp_name = yolo_model.split(".")[0]
elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
    exp_name = yolo_model[0].split(".")[0]
else:  # multiple models after --yolo_model
    exp_name = "ensemble"
exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run if project name exists
(save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

model = DetectMultiBackend(yolo_model, device=device, dnn=dnn)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

# Half
half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
if pt:
    model.model.half() if half else model.model.float()
# print("Done till here!!!")
# # Set Dataloader
vid_path, vid_writer = None, None

# dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
nr_sources = 1
# vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

# initialize DeepSORT
cfg = get_config()
cfg.merge_from_file(config_deepsort)

# Create as many trackers as there are video sources
deepsort_list = []
for i in range(nr_sources):
    deepsort_list.append(
        DeepSort(
            deep_sort_model,
            device,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
        )
    )
outputs = [None] * nr_sources

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names

# Run tracking
model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
dt, seen = [0.0, 0.0, 0.0, 0.0], 0
print("Model Loading successful...")

cameraInUse = []


class VideoCamera(object):
    def __init__(self):
        self.video = cv.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        Thread(target=self.update, args=()).start()

    def __init__(self, cameraId):
        global cameraInUse
        camera = Camera.objects.get(id=cameraId)
        reqStr = "rtsp://username:password@"+camera.ip+":554/user="+camera.username+"_password='"+camera.pwd+"'_channel=channel_number_stream=1.sdp"
        cameraInUse = []
        cameraInUse.append(str(camera.ip))
        self.ip = camera.ip
        print(self.ip)
        print(reqStr)
        if cameraId==1:
            print('if block')
            self.video = cv.VideoCapture(0)
        else:
            print('else block')
            # self.video = cv.VideoCapture(reqStr)
        (self.grabbed, self.frame) = True, np.zeros((75, 75, 3))
        # print(self.grabbed, self.frame)
        Thread(target=self.update, args=[str(camera.ip)]).start()        

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        # image = cv.flip(image, 1)
        _, jpeg = cv.imencode('.jpg', image)
        return jpeg.tobytes()
    @torch.no_grad()
    def update(self, ip):
        # output='inference/output'
        # source='rtsp://username:password@' + str(ip) + ':554/uservid=False,=admin_password=@dmin123_channel=channel_number_stream=1.sdp'
        # path = []
        # agnostic_nms=False
        # augment=False
        # path.append(source)
        # yolo_model='yolov5l.pt'
        # # yolo_model = "C:/Users/WS-AIML-ISE/Desktop/Minor Project - CCTV Research Project/cctv_django-main/cameras/yolov5m-face.pt"
        # deep_sort_model='osnet_x0_25'
        # device=''
        # show_vid=False
        # iou_thres=0.5
        # conf_thres=0.25
        # imgsz=[640, 640]
        # max_det=1000
        # evaluate=False
        # half=False
        # project=WindowsPath('runs/track')
        # visualize=False
        # exist_ok=False
        # update=False
        # save_txt=False
        # save_crop=False
        # dnn=False
        # config_deepsort='C:/Users/WS-AIML-ISE/Desktop/Minor Project - CCTV Research Project/cctv_django-main/cameras/deep_sort.yaml'
        # classes = None
        # device = select_device(device)
        # half &= device.type != 'cpu'
        # allPersons = Person.objects.all()
        # max_person_id = allPersons.aggregate(Max('person_id'))
        # max_person_id = max_person_id['person_id__max']
        # print(max_person_id, "Max Person ID")

        # if type(yolo_model) is str:  # single yolo model
        #     exp_name = yolo_model.split(".")[0]
        # elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
        #     exp_name = yolo_model[0].split(".")[0]
        # else:  # multiple models after --yolo_model
        #     exp_name = "ensemble"
        # exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
        # save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run if project name exists
        # (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # model = DetectMultiBackend(yolo_model, device=device, dnn=dnn)
        # stride, names, pt = model.stride, model.names, model.pt
        # imgsz = check_img_size(imgsz, s=stride)  # check image size
 
        # # Half
        # half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        # if pt:
        #     model.model.half() if half else model.model.float()
        # # print("Done till here!!!")
        # # # Set Dataloader
        # vid_path, vid_writer = None, None

        # dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        # nr_sources = 1
        # # vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

        # # initialize DeepSORT
        # cfg = get_config()
        # cfg.merge_from_file(config_deepsort)

        # # Create as many trackers as there are video sources
        # deepsort_list = []
        # for i in range(nr_sources):
        #     deepsort_list.append(
        #         DeepSort(
        #             deep_sort_model,
        #             device,
        #             max_dist=cfg.DEEPSORT.MAX_DIST,
        #             max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        #             max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
        #         )
        #     )
        # outputs = [None] * nr_sources
        
        # # Get names and colors
        # names = model.module.names if hasattr(model, 'module') else model.names

        # # Run tracking
        # model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
        # dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        # print("Done till here~~~~")
        source='rtsp://username:password@' + str(ip) + ':554/uservid=False,=admin_password=@dmin123_channel=channel_number_stream=1.sdp'
        path = []
        path.append(source)
        camera = Camera.objects.filter(ip=ip).first()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)

        
        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
            print("Primal source"+source)
            print("Primal path"+path[0])
            print("STARTING NEW FRAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("STARTING NEW FRAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            t0 = time.time()
            if ip not in cameraInUse:
                print(cameraInUse)
                (self.grabbed, self.frame) = True, np.zeros((75, 75, 3))
                print("i broke itttt!!!!!! The camera having IP address : "+ip)
                break
            # print(frame_idx)
            # print(path, "******", frame_idx)
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            global visualize
            global seen
            visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, 0, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            # print(pred)
            # break
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p[0])  # to Path
                # # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

                txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                imc = im0[0].copy() if save_crop else im0[0]  # for save_crop
                # print("****************")
                # print(im0[0])
                # print("****************")
                copy_frame = im0[0].copy()
                annotator = Annotator(copy_frame, line_width=2, pil=not ascii)
                # print("****************")
                # print(annotator)
                # print("****************")
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0[0].shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deepsort
                    t4 = time_sync()
                    outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0[0])
                    t5 = time_sync()
                    dt[3] += t5 - t4

                    # draw boxes for visualization
                    if len(outputs[i]) > 0:
                        for j, (output) in enumerate(outputs[i]):
                            bboxes = output[0:4]
                            id = output[4] + max_person_id
                            cls = output[5]
                            conf = output[6]
                            if save_crop or show_vid or True:  # Add bbox to image
                                c = int(cls)  # integer class
                                # label = f'{id:0.0f} {names[c]} {conf:.2f}'
                                label = f'{id:0.0f}'
                                annotator.box_label(bboxes, label, color=colors(c, True))
                                if True:
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    file_storage_location = 'crops' + "/" + names[c] + "/" + f'{id}' + "/" + f'full_person.jpg'
                                    
                                    persons = Person.objects.filter(person_id=id).first()
                                    if persons is not None:
                                        person_instance = persons
                                        if conf > person_instance.conf_score:
                                            cropped_person_img = save_one_box(bboxes, imc, file=WindowsPath(MEDIA_ROOT) / file_storage_location, BGR=True)
                                            person_instance.date_time = datetime.now()
                                            person_instance.person_img = file_storage_location
                                            person_instance.conf_score = conf
                                            person_instance.save()
                                    else:
                                        cropped_person_img = save_one_box(bboxes, imc, file=WindowsPath(MEDIA_ROOT) / file_storage_location, BGR=True)
                                        new_person_instance = Person()
                                        new_person_instance.date_time = datetime.now()
                                        new_person_instance.person_img = file_storage_location
                                        new_person_instance.conf_score = conf
                                        new_person_instance.person_id = id
                                        new_person_instance.camera_id = camera
                                        new_person_instance.save()
                    LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

                else:
                    deepsort_list[i].increment_ages()
                    LOGGER.info('No detections')
                
                # Stream results
                im0[0] = annotator.result()
                self.frame = im0[0]
                self.grabbed = True
                if True:
                    # cv2.imshow(str(p), im0[0])
                    cv2.waitKey(4)  # 1 millisecond
                print("END OF FRAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("END OF FRAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(1/(time.time() - t0))