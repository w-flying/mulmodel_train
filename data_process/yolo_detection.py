import os
import json
import cv2

from ultralytics import YOLO
model = YOLO("yolo11x.pt")
def img_detection(img,img_name):
    results = model(img)[0]
    boxes = results.boxes
    projects = [{model.names[int(x.cls[0])]: float(x.conf[0])} for x in boxes]
    return json.dumps({img_name:projects})




img_files = os.listdir("images")
projects = []
for img_file in img_files:
    img = cv2.imread("images/"+img_file)
    projects.append(img_detection(img,img_file))
    if len(projects) %100==0:
        print(f'已处理到第{len(projects)}图片')
with open("projects_yolo.json","w") as f:
    json.dump(projects,f)





