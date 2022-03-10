# -*- coding: utf-8 -*-
# !pip install -qU bbox-utility
# !pip install -qr requirements.txt  # install
import os
ROOT_DIR  = os.getcwd()
OUTPUT_DIR = ROOT_DIR+'/out'
IMG_DIR = ROOT_DIR + '/images'
CKPT_PATH = 'last.pt' # by @steamedsheep

IMG_SIZE  = 640
CONF      = 0.3
IOU       = 0.40
AUGMENT   = False


import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import glob
import shutil
import sys
sys.path.append(ROOT_DIR)
import time
import json
import torch
from PIL import Image
from utils import *
from bbox.utils import coco2yolo, coco2voc, voc2yolo, voc2coco
from bbox.utils import draw_bboxes, load_image
from bbox.utils import clip_bbox, str2annot, annot2str

os.makedirs(OUTPUT_DIR, exist_ok=True)
colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for i in range(100)]
# colors = [(0, 255, 0) for i in range(5)]
YOLO_DIR = os.getcwd()

def plot_one_box(x, img, score=None, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, "{}:{:.2f}".format(label, score), (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_bboxes(img, bboxes, scores, classes, class_ids, colors = None, show_classes = None, bbox_format = 'yolo', class_name = False, line_thickness = 2):  
     
    image = img.copy()
    show_classes = classes if show_classes is None else show_classes
    colors = (0, 255 ,0) if colors is None else colors
    
    if bbox_format == 'yolo':
        
        for idx in range(len(bboxes)):  
            
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            try:
                score = scores[idx]
            except:
                score = None
            color = colors[cls_id] if type(colors) is list else colors
            
            if cls in show_classes:
            
                x1 = round(float(bbox[0])*image.shape[1])
                y1 = round(float(bbox[1])*image.shape[0])
                w  = round(float(bbox[2])*image.shape[1]/2) #w/2 
                h  = round(float(bbox[3])*image.shape[0]/2)

                voc_bbox = (x1-w, y1-h, x1+w, y1+h)
                plot_one_box(voc_bbox, 
                             image,
                             score= score if score else None,
                             color = color,
                             label = cls if class_name else str(get_label(cls)),
                             line_thickness = line_thickness)
            
    elif bbox_format == 'coco':
        
        for idx in range(len(bboxes)):  
            
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            try:
                score = scores[idx]
            except:
                score = None
            color = colors[cls_id] if type(colors) is list else colors
            
            if cls in show_classes:            
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                w  = int(round(bbox[2]))
                h  = int(round(bbox[3]))

                voc_bbox = (x1, y1, x1+w, y1+h)
                plot_one_box(voc_bbox, 
                             image,
                             score = score if score else None,
                             color = color,
                             label = cls if class_name else str(cls_id),
                             line_thickness = line_thickness)
                
    elif bbox_format == 'voc_pascal':
        
        for idx in range(len(bboxes)):  
            
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            try:
                score = scores[idx]
            except:
                score = None
            color = colors[cls_id] if type(colors) is list else colors
            
            if cls in show_classes: 
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                x2 = int(round(bbox[2]))
                y2 = int(round(bbox[3]))
                voc_bbox = (x1, y1, x2, y2)
                plot_one_box(voc_bbox, 
                             image,
                             score = score if score else None,
                             color = color,
                             label = cls if class_name else str(cls_id),
                             line_thickness = line_thickness)
    else:
        raise ValueError('wrong bbox format')

    return image

def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

def get_imgsize(row):
    row['width'], row['height'] = imagesize.get(row['image_path'])
    return row

def load_model(ckpt_path, conf=0.01, iou=0.50):
    model = torch.hub.load(YOLO_DIR,
                           'custom',
                           path=ckpt_path,
                           source='local',
                           force_reload=True)  # local repo
    model.conf = conf  # NMS confidence threshold
    model.iou  = iou  # NMS IoU threshold
    # model.classes = None   # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    # labels : https://izutsu.aa0.netvolante.jp/pukiwiki/?YOLOv5
    model.classes = [0,2,5,7]   # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image
    return model

def predict(model, img, size=768, augment=False):
    height, width = img.shape[:2]
    results = model(img, size=size, augment=augment)  # custom inference size
    preds   = results.pandas().xyxy[0] # [images, ]
    bboxes  = preds[['xmin','ymin','xmax','ymax']].values
    names = preds['name'].values
    labels = preds['class'].values
    if len(bboxes):
        bboxes  = voc2coco(bboxes,height,width).astype(int)
        confs   = preds.confidence.values
        return bboxes, confs, names, labels
    else:
        return [],[], [], []

def show_img(img, bboxes, confis, names=None, labels=None, bbox_format='yolo', colors=colors):
    img  = draw_bboxes(img = img,
                        bboxes = bboxes,
                        scores = confis,
                        classes = names,
                        class_ids = labels,
                        class_name = True, 
                        colors = colors, 
                        bbox_format = bbox_format,
                        line_thickness = 2)
    return Image.fromarray(img)
    # return img

def do_something(bboxes, confis, names, labels, pil_img):
    global path, log
    t = path.split('.')[0].split('_')[-1]
    car = 0
    truck = 0
    bus = 0
    # bboxesがゼロなら回らない。
    for i in range(len(bboxes)):
        if labels[i]==2:
            car += 1
        elif labels[i]==5:
            bus += 1
        else:
            # == 7
            truck += 1

    all = car+bus+truck
    dic = dict(
        station_name='iwaki',
        timestamp = t,
        car_cnt = car,
        bus = bus,
        truck = truck,
        cnt = all,
        wait_time = car*7 + bus*20 + truck*13
    )
    log.append(dic)
    pil_img.save(OUTPUT_DIR + f'/detect_{t}.jpg')
    return

# read img files
image_paths = glob.glob(IMG_DIR+'/*.jpg')
image_paths += glob.glob(IMG_DIR+'/*.png')
model = load_model(CKPT_PATH, conf=CONF, iou=IOU)
try:
    log = []
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)[...,::-1]
        bboxes, confis, names, labels = predict(model, img, size=IMG_SIZE, augment=AUGMENT)
        pil_img = show_img(img, bboxes, confis, names=names, labels=labels, bbox_format='coco')
        do_something(bboxes, confis, names, labels, pil_img)
    with open(OUTPUT_DIR+'/log.json', mode='w') as f:
        json.dump(log, f)

except:
    print(f'Error : {len(image_paths)} Image file is detected')
    print('We cannot open any images')
