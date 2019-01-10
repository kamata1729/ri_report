import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.models as models

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])
trainset = torchvision.datasets.MNIST(root='./data', 
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=100,
                                            shuffle=True,
                                            num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', 
                                        train=False, 
                                        download=True, 
                                        transform=transform)
testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=100,
                                            shuffle=False, 
                                            num_workers=2)


class MnistNet(nn.Module):
    def __init__(self):
        # (C, H, W)
        super(MnistNet, self).__init__()
        self.conv1 = self.conv_block(1, 32, 3, pad=1) #(1, 28, 28) to (32, 28, 28)
        self.conv2 = self.conv_block(32, 64, 3) #(32, 26, 26) to (64, 24, 24)
        self.conv3 = self.conv_block(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2) 
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*4*4, 128)
        self.fc2 = nn.Linear(128, 10)
        
        
    def forward(self, input):
        h = self.drop(self.pool(self.conv1(input))) #(1, 28, 28) to (32, 28, 28) to (32, 14, 14)
        h = self.drop(self.pool(self.conv2(h))) #(32, 14, 14) to (64, 12, 12) to (64, 6, 6)
        h = self.flatten(self.conv3(h)) #(64, 6, 6) to (128, 4, 4) to flat tensor
        h = self.fc2(self.fc1(h))
        return h
        
    def conv_block(self, in_dim, out_dim, ksize, stride=1, pad=0):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, ksize, stride, pad),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2),
        )
    
    def flatten(self, x):
        bs = x.size()[0]
        return x.view(bs, -1)


def train(max_epoch, model, optim, criterion):
    train_loss_per_epoch = []
    test_loss_per_epoch = []
    for epoch in range(max_epoch):
        scheduler.step()
        train_loss_all = 0
        acc = 0
        model = model.train()
        for imgs, labels in trainloader:
            optim.zero_grad()
            imgs = imgs.cuda()
            labels = labels.cuda()
            pred = model(imgs)
            loss = criterion(pred, labels)
            loss.backward()
            optim.step()
            
            train_loss_all += loss.data
            acc += torch.sum(labels == torch.argmax(pred, dim=1)).cpu().numpy()
            
        train_loss_all = train_loss_all/float(len(trainset))
        train_loss_per_epoch.append(train_loss_all)
        acc = acc/float(len(trainset))
        print("train: epoch: {}, loss: {}, acc: {}".format(epoch, train_loss_all, acc))
        
        acc = 0
        test_loss_all = 0
        model = model.eval()
        for imgs, labels in testloader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            pred = model(imgs)
            loss = criterion(pred, labels)
            
            test_loss_all += loss.data
            acc += torch.sum(labels == torch.argmax(pred, dim=1)).cpu().numpy()
        test_loss_all = test_loss_all / float(len(testset))
        test_loss_per_epoch.append(test_loss_all)
        acc = acc/float(len(testset))
        print("test: epoch: {}, loss: {}, acc: {}".format(epoch, test_loss_all, acc))

        loss_plot(train_loss_per_epoch, test_loss_per_epoch)
        
def loss_plot(train_loss, test_loss):
    epochs = np.arange(len(train_loss))
    plt.clf()
    plt.plot(epochs, train_loss, label='train')
    plt.plot(epochs, test_loss, label='test')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('./loss.png')

def visualize(model):
    first_conv = list(model.children())[0]
    params = list(params.parameters())[0]
    params = params.cpu().numpy()
    params.resize(params.shape[0], params.shape[2], params.shape[3])
    for i, param in enumerate(params):
        param -= param.min()
        img = param.repeat(20, axis=0).repeat(20, axis=1)
        img = np.clip(img, 0, 1)
        img *= 255
        cv2.imwrite("pics/{}.png".format(i), img)

if __name__ == "__main__":
    model = MnistNet().cuda()
    lr = 1e-4
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
    train(30, model, optim, criterion)

    visualize(model)
    
