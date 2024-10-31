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

train_data_loader = Data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)



# 定义优化器
optimizer = torch.optim.Adam(Myvggc.parameters(), lr=0.0005)
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

# # loss 2.531197070799811
# # acc tensor(0.2587, device='cuda:0', dtype=torch.float64)
# loss 2.4123111344486303
# acc tensor(0.3507, device='cuda:0', dtype=torch.float64)
# loss 2.208913961862553
# acc tensor(0.5566, device='cuda:0', dtype=torch.float64)
# loss 2.15421102749819
# acc tensor(0.6091, device='cuda:0', dtype=torch.float64)
# loss 2.1240971805043305
# acc tensor(0.6399, device='cuda:0', dtype=torch.float64)
# loss 2.1112217977556877
# acc tensor(0.6461, device='cuda:0', dtype=torch.float64)
# loss 2.0921462495892036
# acc tensor(0.6681, device='cuda:0', dtype=torch.float64)
# loss 2.090117204533836
# acc tensor(0.6662, device='cuda:0', dtype=torch.float64)
# loss 2.082411708997164
# acc tensor(0.6754, device='cuda:0', dtype=torch.float64)
# loss 2.0720712534954093
# acc tensor(0.6841, device='cuda:0', dtype=torch.float64)
# loss 2.070769820047941
# acc tensor(0.6844, device='cuda:0', dtype=torch.float64)
