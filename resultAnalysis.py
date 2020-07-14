# -*- coding: utf-8 -*-
# 该script通过解析validation输出的各类别检测结果txt文件，从而计算IOU并输出指定置信度的结果以及各类别TP和FP
# 首先用darknet的validation输出各个类别的检测结果：
# ./darknet detector valid KITMoMa/obj.data KITMoMa/yolo-obj.cfg KITMoMa/backup/yolo-obj_last.weights -out ""

import os
import re
import numpy as np
from collections import defaultdict
import pandas as pd
import xml.etree.ElementTree as ET

logPath = "results/"        # Path of Validation results
labelPath_xml = "KITMoMa/xml/"      # Path of original XML label
labelPath_txt = "KITMoMa/jpg/"      # Path of original TXT label

class_names = ['truck', 'excavator', 'wheel loader', 'bulldozer', 'dumper', 'person', 'car']

def calIOU(tup1, tup2):
    """
    计算两个矩形框的交并比。
    :param tup1: (c_x, c_y, w, h) center_x, center_y, width, height
    :param tup2: (c_x, c_y, w, h)
    (x0,y0)矩形左上，（x1,y1）矩形右下：
    :intern param rec1: (x0,y0,x1,y1)
    :intern param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    x0 = tup1[0] - 1./2 * tup1[2]
    y0 = tup1[1] - 1./2 * tup1[3]
    x1 = tup1[0] + 1./2 * tup1[2]
    y1 = tup1[1] + 1./2 * tup1[3]
    rec1 = (x0, y0, x1, y1)
    
    x0 = tup2[0] - 1./2 * tup2[2]
    y0 = tup2[1] - 1./2 * tup2[3]
    x1 = tup2[0] + 1./2 * tup2[2]
    y1 = tup2[1] + 1./2 * tup2[3]
    rec2 = (x0, y0, x1, y1)
    
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    # 两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)


setConfidence = 0.5     # Set confidence threshold value
for className in class_names:
    filname = logPath + className + '.txt'
    
    df_out = pd.DataFrame(columns=['Image','confidence','min_x','min_y','max_x','max_y'])
    
    with open(filname, 'r') as o:
        lines = o.readlines()
        for line in lines:
            tups = line.split(' ')
            img = str(tups[0]) + '.jpg'
            df_out = df_out.append({'Image':img,'confidence':float(tups[1]), 'min_x':float(tups[2]), 'min_y':float(tups[3]), 
                                    'max_x':float(tups[4]), 'max_y':float(tups[5])}, ignore_index=True)
    
    df_sort_conf = df_out.sort_values(by="confidence", ascending=True)
    df_50 = df_sort_conf[(df_sort_conf["confidence"]>=setConfidence)]     #choose result with higher confidence than setConfidence

    outPath = 'KITMoMa/resultAnalysis/result_' + className + '.txt'     # set Output Path of detection result
    with open(outPath, 'w') as out:
        df_50.to_string(out)
    
    
    target_classID = class_names.index(className)
    fp = 0
    tp = 0
    df_fp = pd.DataFrame(columns=['Image','Class_id','min_x','min_y','max_x','max_y', 'iou'])
    
    for row in df_50.itertuples():
        img = getattr(row, 'Image')
        confi_bb = getattr(row, 'confidence')
        minx = getattr(row, 'min_x')
        miny = getattr(row, 'min_y')
        maxx = getattr(row, 'max_x')
        maxy = getattr(row, 'max_y')
        width_bb = maxx - minx
        height_bb = maxy - miny
        
        la_xml = img.split('.')[0] + '.xml'
        la_txt = img.split('.')[0] + '.txt'
        _xml = labelPath_xml + la_xml
        _txt = labelPath_txt + la_txt
        # print(la_xml, la_txt, '\n')
        root = ET.parse(_xml).getroot()
        sz = root.find('size')
        width = float(sz[0].text)   # image width
        height = float(sz[1].text)  # image height
        
        det_cx = minx + 1./2 * width_bb     # center coordinates of detected Bounding Box(Prediction)
        det_cy = miny + 1./2 * height_bb
        
        cx_trans = det_cx / width   # Normalizing center coor. and width & height
        cy_trans = det_cy / height
        width_trans = width_bb / width
        height_trans = height_bb / height
        
        # find corresponding object in label
        with open(_txt) as txt:
            lines = txt.readlines()
            iou_scores = []
            for line in lines:
                tups = line.split(' ')
                classid = int(tups[0])
                cx_la = float(tups[1])      # BB from txt label(Ground Truth)
                cy_la = float(tups[2])
                width_la = float(tups[3])
                height_la = float(tups[4])
                
                # Calculate IOU between BBs from txt label(GT) and from detected results(Pr)
                iou = calIOU([cx_trans,cy_trans,width_trans,height_trans], [cx_la,cy_la,width_la,height_la])
                iou_scores.append(iou)

            if iou_scores:
                max_ind = np.argmax(iou_scores)     # index with max IOU
                max_iou = max(iou_scores)
                class_la = int(lines[max_ind].split(' ')[0])
            else:
                continue
            # for this positive detected object its id is 'obj_id', check whether this number is same as its label
            right_label = (target_classID==class_la)
            # in case the out put class id is not same with label
            if not right_label:
                fp += 1
                df_fp = df_fp.append({'Image':img,'Class_id':class_la, 'min_x':minx, 'min_y':miny, 
                       'max_x':maxx, 'max_y':maxy, 'iou':max_iou}, ignore_index=True)
            else:
                tp += 1
    
    #print("FP of {}(ID:{}) = {1}".format(className, class_names.index(className), fp))
    #print("TP of {}(ID:{}) = {1}\n".format(className, class_names.index(className), tp))
    print("(ID:{}) {}: TP = {},  FP = {}".format(class_names.index(className), className, tp, fp))

    outPath = 'KITMoMa/resultAnalysis/FP_' + className + '.txt'     # set Output Path of FP picture list
    with open(outPath, 'w') as out:
        df_fp.to_string(out)

