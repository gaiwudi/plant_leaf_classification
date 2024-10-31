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
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad_(False)

fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 14),
    # nn.ReLU(),
    # nn.Dropout(0.5),
    # nn.Linear(256, 14),
    nn.LogSoftmax(dim=1))
print(model)
# model=torchvision.models.resnet18(pretrained=True)
## 对训练集的预处理
train_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机长宽比裁剪为224*224
    transforms.RandomHorizontalFlip(),  # 依概率p=0.5水平翻转
    transforms.ToTensor(),  # 转化为张量并归一化至[0-1]
    ## 图像标准化处理
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# 读取图像(从文件夹中直接读取）
train_data_dir = "E:/标准蓝靛果数据集1"

train_data = ImageFolder(train_data_dir, transform=train_data_transforms)

train_data_loader = Data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features,14)
# model = model.to(device)
## 获取vgg16的特征提取层
# model = model.features
# print(model)
# 将vgg16的特征提取层参数冻结，不对其进行更新

# print(model)


## 使用VGG16的特征提取层＋新的全连接层组成新的网络
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         ## 预训练的vgg16的特征提取层
#         self.model = model
#         ## 添加新的全连接层
#         self.classifier = nn.Sequential(
#             nn.Linear(25088, 512),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             # nn.Linear(512, 256),
#             # nn.ReLU(),
#             # nn.Dropout(p=0.5),
#             nn.Linear(512, 14),
#             nn.Softmax(dim=1)
#         )

    # ## 定义网络的向前传播路径
    # def forward(self, x):
    #     x = self.vgg(x)
    #     x = x.view(x.size(0), -1)
    #     output = self.classifier(x)
    #     return output


## 输出我们的网络结构
# model = MyModel()
# print(Myvggc)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)

# ## 可视化网络结构
# hl_graph = hl.build_graph(Myvggc, torch.zeros([1, 3, 224, 224]))
# hl_graph.theme = hl.graph.THEMES["blue"].copy()
# hl_graph
## 将可视化的网路保存为图片,默认格式为pdf
# hl_graph.save("data/chap5/Myvggnet_hl.png", format="png")
## 使用10类猴子的数据集


# ## 对验证集的预处理
# val_data_transforms = transforms.Compose([
#     transforms.Resize(256),  # 重置图像分辨率
#     transforms.CenterCrop(224),  # 依据给定的size从中心裁剪
#     transforms.ToTensor(),  # 转化为张量并归一化至[0-1]
#     ## 图像标准化处理
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])




# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_func = nn.CrossEntropyLoss()  # 损失函数
# 记录训练过程的指标
# history1 = hl.History()
# 使用Canvas进行可视化
# canvas1 = hl.Canvas()
## 对模型进行迭代训练,对所有的数据训练EPOCH轮
for epoch in range(10):
    train_loss_epoch = 0
    val_loss_epoch = 0
    train_corrects = 0
    val_corrects = 0
    ## 对训练数据的迭代器进行迭代计算
    model.train()
    for step, (b_x, b_y) in enumerate(train_data_loader):
        ## 计算每个batch的
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        output = model(b_x)  # CNN在训练batch上的输出
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
    print(epoch)
    print("loss", train_loss)
    print("acc", train_acc)
#
# 0
# loss 2.3068177994965127
# acc tensor(0.4045, device='cuda:0', dtype=torch.float64)
# 1
# loss 1.7542591580076714
# acc tensor(0.6895, device='cuda:0', dtype=torch.float64)
# 2
# loss 1.419770106414839
# acc tensor(0.7607, device='cuda:0', dtype=torch.float64)
# 3
# loss 1.2044490195423192
# acc tensor(0.7811, device='cuda:0', dtype=torch.float64)
# 4
# loss 1.0572735560422688
# acc tensor(0.8026, device='cuda:0', dtype=torch.float64)
# 5
# loss 0.9487439506315771
# acc tensor(0.8121, device='cuda:0', dtype=torch.float64)
# 6
# loss 0.8688980674468024
# acc tensor(0.8240, device='cuda:0', dtype=torch.float64)
# 7
# loss 0.81077679554162
# acc tensor(0.8181, device='cuda:0', dtype=torch.float64)
# 8
# loss 0.7547911873442589
# acc tensor(0.8367, device='cuda:0', dtype=torch.float64)
