import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt

import seaborn as sns
import hiddenlayer as hl


import torch.nn as nn
from torch.optim import SGD,Adam
import torch.utils.data as Data
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder
mobilenetv2 = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
print(mobilenetv2)

mobilenetv2 = mobilenetv2.features
# 将特征提取层参数冻结，不对其进行更新
for param in mobilenetv2.parameters():
    param.requires_grad_(False)


## 使用的特征提取层＋新的全连接层组成新的网络
class Mymobilenetv2(nn.Module):
    def __init__(self):
        super(Mymobilenetv2, self).__init__()
        ## 预训练的特征提取层
        self.mobilenetv2 = mobilenetv2
        ## 添加新的全连接层
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(512, 14),
            nn.Softmax(dim=1)
        )

    ## 定义网络的向前传播路径
    def forward(self, x):
        x = self.mobilenetv2(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


## 输出我们的网络结构
Mymobilenet_v2 = Mymobilenetv2()
# print(Myvggc)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Mymobilenet_v2 = Mymobilenet_v2.to(device)

## 对训练集的预处理
train_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(32),  # 随机长宽比裁剪为224*224
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

train_data_loader = Data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

# print("数据集的lable：", train_data.targets)
# 定义优化器
optimizer = torch.optim.Adam(Mymobilenet_v2.parameters(), lr=0.0001)
loss_func = nn.CrossEntropyLoss()  # 损失函数
# 记录训练过程的指标
history1 = hl.History()
# 使用Canvas进行可视化
canvas1 = hl.Canvas()
## 对模型进行迭代训练,对所有的数据训练EPOCH轮
for epoch in range(30):
    train_loss_epoch = 0
    val_loss_epoch = 0
    train_corrects = 0
    val_corrects = 0
    ## 对训练数据的迭代器进行迭代计算
    Mymobilenet_v2.train()
    for step, (b_x, b_y) in enumerate(train_data_loader):
        ## 计算每个batch的
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        output = Mymobilenet_v2(b_x)  # CNN在训练batch上的输出
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
torch.save(Mymobilenet_v2,"data/Mymobilenet_v2.pkl")
## 导入保存的模型
Mymobilenetv2 = torch.load("data/Mymobilenet_v2.pkl")
Mymobilenetv2
loss 2.590493528553516
# acc tensor(0.1692, device='cuda:0', dtype=torch.float64)
# loss 2.4919442339439613
# acc tensor(0.3114, device='cuda:0', dtype=torch.float64)
# loss 2.4330584636313377
# acc tensor(0.3642, device='cuda:0', dtype=torch.float64)
# loss 2.4061622360538197
# acc tensor(0.3814, device='cuda:0', dtype=torch.float64)
# loss 2.390027254578695
# acc tensor(0.3897, device='cuda:0', dtype=torch.float64)
# loss 2.383353437578058
# acc tensor(0.3939, device='cuda:0', dtype=torch.float64)