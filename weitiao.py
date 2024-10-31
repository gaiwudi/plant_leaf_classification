## 导入本章所需要的模块
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt

import seaborn as sns
import hiddenlayer as hl

import torch
import torch.nn as nn
from torch.optim import SGD,Adam
import torch.utils.data as Data
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder
vgg16 = models.vgg16(pretrained=True)
# print(vgg16)
## 获取vgg16的特征提取层
vgg = vgg16.features
# 将vgg16的特征提取层参数冻结，不对其进行更新
for param in vgg.parameters():
    param.requires_grad_(False)


## 使用VGG16的特征提取层＋新的全连接层组成新的网络
class MyVggModel(nn.Module):
    def __init__(self):
        super(MyVggModel, self).__init__()
        ## 预训练的vgg16的特征提取层
        self.vgg = vgg
        ## 添加新的全连接层
        self.classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 14),
            nn.Softmax(dim=1)
        )

    ## 定义网络的向前传播路径
    def forward(self, x):
        x = self.vgg(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


## 输出我们的网络结构
Myvggc = MyVggModel()
# print(Myvggc)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Myvggc = Myvggc.to(device)

# ## 可视化网络结构
# hl_graph = hl.build_graph(Myvggc, torch.zeros([1, 3, 224, 224]))
# hl_graph.theme = hl.graph.THEMES["blue"].copy()
# hl_graph
## 将可视化的网路保存为图片,默认格式为pdf
# hl_graph.save("data/chap5/Myvggnet_hl.png", format="png")
## 使用10类猴子的数据集
## 对训练集的预处理
train_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机长宽比裁剪为224*224
    transforms.RandomHorizontalFlip(),  # 依概率p=0.5水平翻转
    transforms.ToTensor(),  # 转化为张量并归一化至[0-1]
    ## 图像标准化处理
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

## 对验证集的预处理
val_data_transforms = transforms.Compose([
    transforms.Resize(256),  # 重置图像分辨率
    transforms.CenterCrop(224),  # 依据给定的size从中心裁剪
    transforms.ToTensor(),  # 转化为张量并归一化至[0-1]
    ## 图像标准化处理
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 读取图像(从文件夹中直接读取）
train_data_dir = "E:/标准蓝靛果数据集1"

train_data = ImageFolder(train_data_dir, transform=train_data_transforms)

train_data_loader = Data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)

# print("数据集的lable：", train_data.targets)

# ## 读取图像（从分好的文件中读取）
# train_data_dir = "data/chap6/10-monkey-species/training"
# train_data = ImageFolder(train_data_dir, transform=train_data_transforms)
# train_data_loader = Data.DataLoader(train_data,batch_size=32,
#                                     shuffle=True,num_workers=2)
# ## 读取验证集
# val_data_dir = "data/chap6/10-monkey-species/validation"
# val_data = ImageFolder(val_data_dir, transform=val_data_transforms)
# val_data_loader = Data.DataLoader(val_data,batch_size=32,
#                                   shuffle=True,num_workers=2)
# print("训练集样本数:",len(train_data.targets))
# print("验证集样本数:",len(val_data.targets))


# ##  获得一个batch的数据
# for step, (b_x, b_y) in enumerate(train_data_loader):
#     if step > 0:
#         break
# for i, item in enumerate(train_data_loader):
#         print('i:', i)
#         data, label = item
#         if torch.cuda.is_available():
#             data = data.cuda()
#             label = label.cuda()
#         print('data:', data)
#         print('label:', label)

## 输出训练图像的尺寸和标签的尺寸
# print(b_x.shape)
# print(b_y.shape)
# len(val_data.targets)
# len(train_data.targets)
# # print(len(val_data_loader))
# print(len(train_data_loader))
## 可视化训练集其中一个batch的图像
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
# plt.figure(figsize=(12,6))
# for ii in np.arange(len(b_y)):
#     plt.subplot(4,8,ii+1)
#     image = b_x[ii,:,:,:].numpy().transpose((1, 2, 0))
#     image = std * image + mean
#     image = np.clip(image, 0, 1)
#     plt.imshow(image)
#     plt.title(b_y[ii].data.numpy())
#     plt.axis("off")
# plt.subplots_adjust(hspace = 0.3)
##  获得一个batch的数据
# for step, (b_x, b_y) in enumerate(val_data_loader):
#     if step > 0:
#         break

## 输出训练图像的尺寸和标签的尺寸
# print(b_x.shape)
# print(b_y.shape)

## 可视化验证集其中一个batch的图像
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
# plt.figure(figsize=(12,6))
# for ii in np.arange(len(b_y)):
#     plt.subplot(4,8,ii+1)
#     image = b_x[ii,:,:,:].numpy().transpose((1, 2, 0))
#     image = std * image + mean
#     image = np.clip(image, 0, 1)
#     plt.imshow(image)
#     plt.title(b_y[ii].data.numpy())
#     plt.axis("off")
# plt.subplots_adjust(hspace = 0.3)
# 定义优化器
optimizer = torch.optim.Adam(Myvggc.parameters(), lr=0.0003)
loss_func = nn.CrossEntropyLoss()  # 损失函数
# 记录训练过程的指标
history1 = hl.History()
# 使用Canvas进行可视化
canvas1 = hl.Canvas()
## 对模型进行迭代训练,对所有的数据训练EPOCH轮
for epoch in range(10):
    train_loss_epoch = 0
    val_loss_epoch = 0
    train_corrects = 0
    val_corrects = 0
    ## 对训练数据的迭代器进行迭代计算
    Myvggc.train()
    for step, (b_x, b_y) in enumerate(train_data_loader):
        ## 计算每个batch的
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        output = Myvggc(b_x)  # CNN在训练batch上的输出
        loss = loss_func(output, b_y)  # 交叉熵损失函数
        pre_lab = torch.argmax(output, 1)
        optimizer.zero_grad()  # 每个迭代步的梯度初始化为0
        loss.backward()  # 损失的后向传播，计算梯度
        optimizer.step()  # 使用梯度进行优化
        train_loss_epoch += loss.item() * b_x.size(0)
        train_corrects += torch.sum(pre_lab == b_y.data)
    ## 计算一个epoch的损失和精度
    train_loss = train_loss_epoch / len(train_data.targets)
    train_acc = train_corrects.double() / len(train_data.targets)
    print("loss", train_loss)
    print("acc", train_acc)

    # ## 计算在验证集上的表现
    # Myvggc.eval()
    # for step, (val_x, val_y) in enumerate(val_data_loader):
    #     output = Myvggc(val_x)
    #     loss = loss_func(output, val_y)
    #     pre_lab = torch.argmax(output, 1)
    #     val_loss_epoch += loss.item() * val_x.size(0)
    #     val_corrects += torch.sum(pre_lab == val_y.data)
    # ## 计算一个epoch的损失和精度
    # val_loss = val_loss_epoch / len(val_data.targets)
    # val_acc = val_corrects.double() / len(val_data.targets)

    ## 保存每个epoch上的输出loss和acc  val_acc=val_acc.item()
    history1.log(epoch, train_loss=train_loss,
                 train_acc=train_acc.item(),
                 )
    # 可视网络训练的过程 , history1["val_loss"], history1["val_acc"]
    with canvas1:
        canvas1.draw_plot([history1["train_loss"]])
        canvas1.draw_plot([history1["train_acc"]])
## 保存模型  val_loss=val_loss,
torch.save(Myvggc,"data/chap6/Myvggc.pkl")
## 导入保存的模型
Myvggc2 = torch.load("data/chap6/Myvggc.pkl")
Myvggc2
# loss 2.392939238258869
# acc tensor(0.3616, device='cuda:0', dtype=torch.float64)
# loss 2.238429603342376
# acc tensor(0.5201, device='cuda:0', dtype=torch.float64)
# loss 2.174571994412152
# acc tensor(0.5819, device='cuda:0', dtype=torch.float64)
# loss 2.1510829398397764
# acc tensor(0.6033, device='cuda:0', dtype=torch.float64)
# loss 2.1361349118238238
# acc tensor(0.6166, device='cuda:0', dtype=torch.float64)
# loss 2.114018202654888
# acc tensor(0.6397, device='cuda:0', dtype=torch.float64)
# loss 2.1091203434618913
# acc tensor(0.6448, device='cuda:0', dtype=torch.float64)
# loss 2.094565454314899
# acc tensor(0.6579, device='cuda:0', dtype=torch.float64)
# loss 2.0960080642231627
# acc tensor(0.6572, device='cuda:0', dtype=torch.float64)
# loss 2.0840158949697636
# acc tensor(0.6697, device='cuda:0', dtype=torch.float64