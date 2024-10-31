# ## 导入本章所需要的模块
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import requests
# import cv2
#
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torchvision import models
# from torchvision import transforms
# from PIL import Image
#
# import sklearn
# import torchvision
#
#
#
#
# import torch.utils.data as Data
# # import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
#
#
#
# train_data_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(224),  # 随机长宽比裁剪为224*224
#     transforms.RandomHorizontalFlip(),  # 概率p=0.5水平翻转
#     transforms.ToTensor(),  # 转化为张量并归一化至[0-1]
#     # 图像标准化处理
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])
#
# # 读取图像
# train_data_dir = "E:/标准蓝靛果数据集"
# train_data = ImageFolder(train_data_dir, transform=train_data_transforms)
# train_data_loader = Data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=1)
# # print("数据集的lable：", train_data.targets)
# # 导入预训练好的VGG16网络
# vgg16 = models.vgg16(pretrained=True)
# print(vgg16)
# ## 读取一张图片,并对其进行可视化
# im = Image.open("E:/标准蓝靛果数据集/A1品种/1.jpg")
# imarray = np.asarray(im) / 255.0
# plt.figure()
# plt.imshow(imarray)
# plt.show()
# imarray.shape
# ## 对一张图像处理为vgg16网络可以处理的形式
# data_transforms = transforms.Compose([
#     transforms.Resize((224,224)),# 重置图像分辨率
#     transforms.ToTensor(),# 转化为张量并归一化至[0-1]
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# input_im = data_transforms(im).unsqueeze(0)
# print("input_im.shape:",input_im.shape)
# ## 使用钩子获取分类层的2个特征
# ## 定义一个辅助函数，来获取指定层名称的特征
# activation = {} ## 保存不同层的输出
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook
# ## 获取中间的卷积后的图像特征
# vgg16.eval()
# ##  第四层，经过第一次最大值池化
# vgg16.features[4].register_forward_hook(get_activation("maxpool1"))
# _ = vgg16(input_im)
# maxpool1 = activation["maxpool1"]
# print("获取特征的尺寸为:",maxpool1.shape)
# ## 对中间层进行可视化,可视化64个特征映射
# plt.figure(figsize=(11,6))
# for ii in range(maxpool1.shape[1]):
#     ## 可视化每张手写体
#     plt.subplot(6,11,ii+1)
#     plt.imshow(maxpool1.data.numpy()[0,ii,:,:],cmap="gray")
#     plt.axis("off")
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
# plt.show()
# ## 获取更深层次的卷积后的图像特征
# vgg16.eval()
# vgg16.features[21].register_forward_hook(get_activation("layer21_conv"))
# _ = vgg16(input_im)
# layer21_conv = activation["layer21_conv"]
# print("获取特征的尺寸为:",layer21_conv.shape)
# ## 对中间层进行可视化,只可视化前72个特征映射
# plt.figure(figsize=(12,6))
# for ii in range(72):
#     ## 可视化每张手写体
#     plt.subplot(6,12,ii+1)
#     plt.imshow(layer21_conv.data.numpy()[0,ii,:,:],cmap="gray")
#     plt.axis("off")
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
# plt.show()
# ## 获取vgg模型训练时对应的1000类的类别标签
# LABELS_URL = "https://s3.amazonaws.com/outcome-blog/imagenet/labels.json"
# # 从网页链接中获取类别标签
# response = requests.get(LABELS_URL)
# labels = {int(key): value for key, value in response.json().items()}
# ## 使用VGG16网络预测图像的种类
# vgg16.eval()
# im_pre = vgg16(input_im)
# ## 计算预测top-5的可能性
# softmax = nn.Softmax(dim=1)
# im_pre_prob = softmax(im_pre)
# prob,prelab = torch.topk(im_pre_prob,5)
# prob = prob.data.numpy().flatten()
# prelab = prelab.numpy().flatten()
# for ii,lab in enumerate(prelab):
#     print("index: ", lab ," label: ",labels[lab]," ||",prob[ii])

import torch
model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
model.eval()
# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)





# Download an example image from the pytorch website
import urllib

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
# Download ImageNet labels
# ! wget
# https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
