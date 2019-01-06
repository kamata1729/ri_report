
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

test_size = 10000
train_size = 60000
mnist = datasets.fetch_mldata('MNIST original')
train_X, test_X, train_Y, test_Y = train_test_split(mnist.data / 255, 
                                                        mnist.target, 
                                                        test_size=test_size, 
                                                        train_size=train_size)
del mnist
train_Y = train_Y.astype(np.int32)
train_Y = np.eye(10)[train_Y]
test_Y = test_Y.astype(np.int32)
test_Y = np.eye(10)[test_Y]

class Sigmoid:
    def __init__(self):
        self.y = None
        self.use_bp = False #BP対象か
        
    # 順伝播
    def forward(self, x):
        y = 1 / (1+np.exp(-x)) 
        self.y = y
        return y
    
    # 逆伝播
    def backward(self, output):
        return self.y*output*(1-self.y)

class ReLU:
    def __init__(self):
        self.x = None
        self.use_bp = False #BP対象か
    
    # 順伝播
    def forward(self, x):
        self.x = x
        return x * (x > 0)  
    
    # 逆伝播
    def backward(self, output):
        return output * (self.x > 0)  

class Softmax:
    def __init__(self):
        self.x = None
        self.y = None
        self.use_bp = False #BP対象か
        
    def forward(self, x):
        self.x = x
        tmp = np.exp(x - x.max(axis=1, keepdims=True))
        y = tmp/np.sum(tmp, axis=1, keepdims=True)
        self.y = y
        return y

class Linear:
    def __init__(self, in_dim, out_dim):
        hp = 0.01
        #初期値はwはrandom, bは0に設定
        self.w = hp*np.random.randn(in_dim, out_dim)
        self.b = np.zeros(out_dim)
        
        self.x = None
        # 勾配を保存しておくインスタンス
        self.dw = None
        self.db = None
        self.delta = None
        
        self.use_bp = True #BP対象か

    # 順伝播
    def forward(self, x):
        self.x = x
        # output = XW + b
        out = x@self.w + self.b
        
        return out
    
    # 逆伝播
    def backward(self, delta):
        self.dw =  self.x.T@delta
        ones = np.ones(len(self.x))
        self.db = ones@delta
        out = delta@self.w.T
        return out

class SGD():
    def __init__(self, lr=0.01):
        self.lr = lr
        self.model = None
    
    def set_model(self, model):
        self.model = model
    
    def update(self):
        for layer in self.model.layers:
            if layer.use_bp:
                layer.w -= self.lr * layer.dw
                layer.b -=  self.lr * layer.db

class NN():
    def __init__(self, layers, transform=None):
        # layersは各層が連なった配列
        self.layers = layers
        self.transform = transform
        self.t = None
        
    def calc_loss(self, y, t):
        return np.sum(-t*np.log(y+1e-6)) / len(t)
        
    # tは教師データ
    def forward(self, input, t):   
        self.t = t
        # pytorchリスペクトでtransform
        if self.transform:
            input = self.transform(input)
            
        #最初の層のyはinputに等しいとする
        self.y = input
        # 順に順伝播計算をして行く
        for layer in self.layers:
            self.y = layer.forward(self.y)
        
        # この時点でyにはNNの出力が入っている
        loss =  self.calc_loss(self.y, t)
        self.loss = loss
        return loss
    
    def backward(self):
        out = (self.y - self.t) / len(self.layers[-1].x)
        
        # 順伝播で計算したdW, dbを元に逆伝播計算する
        for layer in self.layers[-2::-1]:
            out =  layer.backward(out)    

class RandomNoise(object):
    def __init__(self, noise_rate):
        self.noise_rate = noise_rate
        
    def __call__(self, x):
        """
        x :(batchsize, 784)
        """
        noise_tf = np.random.rand(x.shape[0], x.shape[1]) < self.noise_rate
        return x - x * noise_tf + np.random.rand(x.shape[0], x.shape[1])*noise_tf

def train(model, optim, max_epoch=5, bsize=50):
    train_loss_per_epoch = []
    test_loss_per_epoch = []
    
    for epoch in range(max_epoch):
        """ train """
        pred = np.array([])
        train_loss_all = 0
        
        # 一応シャッフルしておく
        perm = np.random.permutation(train_size)
        
        for i in range(0, train_size, bsize):
            X = train_X[perm[i:i+bsize]]
            Y = train_Y[perm[i:i+bsize]]
            
            train_loss = model.forward(X, Y)
            train_loss_all += train_loss*len(Y)
            model.backward()
            optim.update()
            pred = np.append(pred, np.argmax(model.y, axis=1)).astype(np.int32)
        
        pred = np.eye(10)[pred]
        score = np.sum(pred * train_Y[perm])/train_size
        train_loss = train_loss_all/train_size
        train_loss_per_epoch.append(train_loss)
        print("train: epoch: {}, loss: {}, score: {}".format(epoch, train_loss, score))
        
        """ test """
        pred = np.array([])
        test_loss_all = 0
        for i in range(0, test_size, bsize):
            X = test_X[i:i+bsize]
            Y = test_Y[i:i+bsize]
            test_loss = model.forward(X, Y) * len(Y)
            test_loss_all += test_loss
            pred = np.append(pred, np.argmax(model.y, axis=1)).astype(np.int32)
            
        pred = np.eye(10)[pred]
        score = np.sum(pred * test_Y)/test_size
        test_loss = test_loss_all/test_size
        test_loss_per_epoch.append(test_loss)
        print("test: epoch: {}, loss: {}, score: {}".format(epoch, test_loss, score))
    
    plt.figure()
    plt.plot(np.arange(max_epoch), train_loss_per_epoch, label='train_loss')
    plt.plot(np.arange(max_epoch), test_loss_per_epoch, label='test_loss')
    plt.legend()
    plt.savefig("loss.png")
    plt.show()

if __name__ == '__main__':
    optim = SGD(lr=0.5)
    model = NN([Linear(784, 1000),
                ReLU(),
                Linear(1000, 1000),
                ReLU(),
                Linear(1000, 10),
                Softmax()],
           transform=RandomNoise(noise_rate=0.25))
    optim.set_model(model)
    train(model=model, optim=optim, max_epoch=5, bsize=50)
