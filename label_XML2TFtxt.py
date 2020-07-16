import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = []
classes = ["truck", "excavator", "wheel loader", "bulldozer", "dumper", "person", "car"]  ##change to your own classes

def convert_annotation(image_add):
    image_add = os.path.split(image_add)[1]     # image_add = 00001.jpg
    image_add = image_add[0:image_add.find('.',1)]  # image_add = 00001

    in_file = open('/home/niuzhuo/YOLOv4/yolov4-tflite/data/KITMoMa/xml/' + image_add + '.xml')

    tree=ET.parse(in_file)
    root = tree.getroot()

    if root.find('size'):
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        if w==0 or h == 0:
            print("Wrong! width or height is 0:  "+image_add)
            os.remove("/home/niuzhuo/YOLOv4/yolov4-tflite/data/KITMoMa/xml/"+image_add+".xml")
            return


        imgPath = "/home/niuzhuo/YOLOv4/yolov4-tflite/data/KITMoMa/jpg/" + image_add + ".jpg"
        out_file.write(str(imgPath) + " ")

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text

            if cls not in classes or int(difficult)==1:
                continue

            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')

            b = (float(xmlbox.find('xmin').text), 
                    float(xmlbox.find('ymin').text), 
                    float(xmlbox.find('xmax').text), 
                    float(xmlbox.find('ymax').text), 
                    int(cls_id))

            out_file.write(",".join([str(a) for a in b]) + " ")

        out_file.write("\n")
        print("Done!    " + imgPath)


    else:
        print("Error! xml need size:  "+image_add)
        #os.remove("G:/set/"+image_add+".jpg")
        os.remove("/home/niuzhuo/YOLOv4/darknet/KITMoMa/xml/"+image_add+".xml")

out_file = open('/home/niuzhuo/YOLOv4/yolov4-tflite/data/KITMoMa/tfLabel/train.txt', 'w')
image_adds = open("/home/niuzhuo/YOLOv4/yolov4-tflite/data/KITMoMa/train.txt")
for image_add in image_adds:
    convert_annotation(image_add)
