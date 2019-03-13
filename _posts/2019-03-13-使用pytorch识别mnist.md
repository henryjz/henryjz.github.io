---
layout: post
title:  "使用pytorch识别mnist"
date: 2019-03-13 21:35:10 +0800
categories: reproduction
tags: pytorch
author: Henryzhou
---


> 转载自：https://github.com/zergtant/pytorch-handbook/blob/master/chapter3/3.2-mnist.ipynb

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
torch.__version__
```




    '1.0.1.post2'



# 3.2  MNIST数据集手写数字识别

## 3.2.1  数据集介绍
MNIST 包括6万张28x28的训练样本，1万张测试样本，很多教程都会对它”下手”几乎成为一个 “典范”，可以说他就是计算机视觉里面的Hello World。所以我们这里也会使用MNIST来进行实战。

我们在介绍卷积神经网络的时候说到过LeNet-5，LeNet-5之所以强大就是因为在当时的环境下将MNIST数据的识别率提高到了99%，这里我们也自己从头搭建一个卷积神经网络，也达到99%的准确率

## 3.2.2 手写数字识别
首先，我们定义一些超参数


```python
BATCH_SIZE=512 #大概需要2G的显存
EPOCHS=20 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
```

因为Pytorch里面包含了MNIST的数据集，所以我们这里直接使用即可。
如果第一次执行会生成data文件夹，并且需要一些时间下载，如果以前下载过就不会再次下载了

由于官方已经实现了dataset，所以这里可以直接使用DataLoader来对数据进行读取


```python
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, 
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to data/MNIST/raw/train-images-idx3-ubyte.gz


    100.1%
    
    Extracting data/MNIST/raw/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to data/MNIST/raw/train-labels-idx1-ubyte.gz


    113.5%
    
    Extracting data/MNIST/raw/train-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to data/MNIST/raw/t10k-images-idx3-ubyte.gz


    100.4%
    
    Extracting data/MNIST/raw/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to data/MNIST/raw/t10k-labels-idx1-ubyte.gz


    180.4%
    
    Extracting data/MNIST/raw/t10k-labels-idx1-ubyte.gz
    Processing...
    Done!


测试集


```python
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)
```

下面我们定义一个网络，网络包含两个卷积层，conv1和conv2，然后紧接着两个线性层作为输出，最后输出10个维度，这10个维度我们作为0-9的标识来确定识别出的是那个数字

在这里建议大家将每一层的输入和输出都显作为注释标注出来，这样后面阅读代码的会方便很多


```python
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1=nn.Conv2d(1,10,5) # 10, 24x24
        self.conv2=nn.Conv2d(10,20,3) # 128, 10x10
        self.fc1 = nn.Linear(20*10*10,500)
        self.fc2 = nn.Linear(500,10)
    def forward(self,x):
        in_size = x.size(0)
        out = self.conv1(x) #24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  #12
        out = self.conv2(out) #10
        out = F.relu(out)
        out = out.view(in_size,-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out
```

我们实例化一个网络，实例化后使用.to方法将网络移动到GPU

优化器我们也直接选择简单暴力的Adam


```python
model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())
```

下面定义一下训练的函数，我们将训练的所有操作都封装到这个函数中


```python
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
```

测试的操作也一样封装成一个函数


```python
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```

下面开始训练，封装起来的好处这里就体现出来了，只要谢两行就可以了


```python
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
```

    Test set: Average loss: 0.0383, Accuracy: 9900/10000 (99%)
    
    Train Epoch: 20 [14848/60000 (25%)]	Loss: 0.002005
    Train Epoch: 20 [30208/60000 (50%)]	Loss: 0.005618
    Train Epoch: 20 [45568/60000 (75%)]	Loss: 0.003705
    
    Test set: Average loss: 0.0415, Accuracy: 9893/10000 (99%)



我们看一下结果，准确率99%，没问题

如果你的模型连MNIST都搞不定，那么你的模型没有任何的价值

如果你的模型搞定了MNIST，那么你的模型也可能没有任何的价值

MNIST是一个很简单的数据集，但是因为他的局限性只能作为研究来使用，对于实际应用中带来的价值非常有限，但是通过这个例子，我们可以完全了解一个实际项目的工作流程

我们找到数据集，对数据做预处理，定义我们的模型，调整超参数，测试训练，再通过训练结果对超参数进行调整或者对模型进行调整。

并且通过这个实战我们已经有了一个很好的模板，以后的项目都可以以这个模板为样例
