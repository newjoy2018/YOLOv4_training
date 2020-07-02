# YOLOv4 Training on Colab

## 1. Set GPU mode

## 2. load Google Drive

```shell
# Set GPU mode

# load Google Drive
from google.colab import drive
drive.mount('/content/drive')
```
## 3.Clone darknet from Github to My Drive
```shell
%cd drive/My\ Drive
!git clone https://github.com/AlexeyAB/darknet.git
```
## 4.Modify the **Makefile** and **Compile**
```
%cd darknet

!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile

!make
```
## 5. Load dataset and configuration
```
%cd ..
!mv KITMoMa/ darknet
%cd darknet
```
## 6. Start training
```
!./darknet detector train KITMoMa/obj.data KITMoMa/yolo-obj.cfg KITMoMa/backup/yolo-obj_last.weights -dont_show -map
```
