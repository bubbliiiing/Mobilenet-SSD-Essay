## Mobilenet-SSD：轻量级目标检测模型在Keras当中的实现（论文版）
---

### 嘟嘟嘟为什么要再弄一个版本的Mobilenet-SSD
之前实现了一个版本的mobilenet-SSD，有小伙伴告诉我说这个不是原版的Mobilenet-ssd的结构，然后我去网上查了一下，好像还真不是，原版的Mobilenet-ssd不利用38x38的特征层进行回归预测和分类预测，因此我就制作了这个版本，填一下坑。

### 目录
1. [所需环境 Environment](#所需环境)
2. [文件下载 Download](#文件下载)
3. [训练步骤 How2train](#训练步骤)
4. [参考资料 Reference](#Reference)

### 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  
### 文件下载
训练所需的essay_mobilenet_ssd_weights可以在百度云下载。  
链接: https://pan.baidu.com/s/1YlD2KAZTv581YSWkC3_ZGg  
提取码: dvau  
### 训练步骤
1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4、在训练前利用voc2ssd.py文件生成对应的txt。  
5、再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。  
```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6、就会生成对应的2007_train.txt，每一行对应其图片位置及其真实框的位置。  
7、在训练前需要修改model_data里面的voc_classes.txt文件，需要将classes改成你自己的classes。  
8、运行train.py即可开始训练。  

### mAP目标检测精度计算更新
更新了get_gt_txt.py、get_dr_txt.py和get_map.py文件。  
get_map文件克隆自https://github.com/Cartucho/mAP  
具体mAP计算过程可参考：https://www.bilibili.com/video/BV1zE411u7Vw

### Reference
https://github.com/pierluigiferrari/ssd_keras  
https://github.com/kuhung/SSD_keras  
https://github.com/ruinmessi/RFBNet
