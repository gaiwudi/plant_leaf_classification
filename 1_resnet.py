import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
import torch.nn as nn
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)
# print(model)
# model.eval()
# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
# Download an example image from the pytorch website
# import urllib
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)
# sample execution (requires torchvision)
# from PIL import Image
# from torchvision import transforms
# input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
train_data_dir = "E:/标准蓝靛果数据集1"

train_data = ImageFolder(train_data_dir, transform=preprocess)

train_data_loader = Data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) #
if torch.cuda.is_available():
    # train_data_loader = train_data_loader.to('cuda')
    model.to('cuda')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# with torch.no_grad():
#     output = model(input_batch)
# # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)
# # Download ImageNet labels
# !wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
# # Read the categories
# with open("imagenet_classes.txt", "r") as f:
#     categories = [s.strip() for s in f.readlines()]
# # Show top categories per image
# top5_prob, top5_catid = torch.topk(probabilities, 5)
# for i in range(top5_prob.size(0)):
#     print(categories[top5_catid[i]], top5_prob[i].item())




# with torch.no_grad():
#     output = model(train_data_loader)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)


if hasattr(torch.cuda, 'empty_cache'):
	torch.cuda.empty_cache()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_func = nn.CrossEntropyLoss()  # 损失函数

## 对模型进行迭代训练,对所有的数据训练EPOCH轮
for epoch in range(30):
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
    print("loss", train_loss)
    print("acc", train_acc)

# loss 0.5404030996186368
# acc tensor(0.8812, device='cuda:0', dtype=torch.float64)
# loss 0.04434591889962813
# acc tensor(0.9858, device='cuda:0', dtype=torch.float64)
# loss 0.032803973475083535
# acc tensor(0.9900, device='cuda:0', dtype=torch.float64)
# loss 0.04271617389467109
# acc tensor(0.9874, device='cuda:0', dtype=torch.float64)
# loss 0.04023838830769272
# acc tensor(0.9874, device='cuda:0', dtype=torch.float64)
# loss 0.0247265678965029
# acc tensor(0.9932, device='cuda:0', dtype=torch.float64)
# loss 0.03242833919633406
# acc tensor(0.9893, device='cuda:0', dtype=torch.float64)
# loss 0.022080868278963845
# acc tensor(0.9928, device='cuda:0', dtype=torch.float64)
# loss 0.0108254196370933
# acc tensor(0.9961, device='cuda:0', dtype=torch.float64)
# loss 0.021969202963158897
# acc tensor(0.9931, device='cuda:0', dtype=torch.float64)
# loss 0.023017009508404396
# acc tensor(0.9926, device='cuda:0', dtype=torch.float64)
# loss 0.05918133923719298
# acc tensor(0.9822, device='cuda:0', dtype=torch.float64)
# loss 0.04488309365098305
# acc tensor(0.9845, device='cuda:0', dtype=torch.float64)
# loss 0.016977153878222958
# acc tensor(0.9947, device='cuda:0', dtype=torch.float64)
# loss 0.013680683662024773
# acc tensor(0.9954, device='cuda:0', dtype=torch.float64)
# loss 0.005694897881960712
# acc tensor(0.9987, device='cuda:0', dtype=torch.float64)
# loss 0.014326882650039422
# acc tensor(0.9951, device='cuda:0', dtype=torch.float64)
# loss 0.012519940872733197
# acc tensor(0.9967, device='cuda:0', dtype=torch.float64)
# loss 0.017984897450182582
# acc tensor(0.9945, device='cuda:0', dtype=torch.float64)
# loss 0.040904494086778424
# acc tensor(0.9876, device='cuda:0', dtype=torch.float64)
# loss 0.04107667093042074
# acc tensor(0.9868, device='cuda:0', dtype=torch.float64)
# loss 0.014061803121767087
# acc tensor(0.9952, device='cuda:0', dtype=torch.float64)
# loss 0.018882095804700433
# acc tensor(0.9949, device='cuda:0', dtype=torch.float64)
