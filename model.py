import torch.nn as nn
import torch.nn.functional as F


class SCAE(nn.Module):
    
    def __init__(self, device='cpu'):
        super(SCAE, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 100, 10, stride=2)
        self.pool1 = nn.MaxPool2d(2, stride=1, return_indices=True)
        self.conv2 = nn.Conv2d(100, 200, 10, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=1, return_indices=True)
        self.conv3 = nn.Conv2d(200, 500, 10, stride=1)
        self.pool3 = nn.MaxPool2d(2, stride=1, return_indices=True)
        self.conv4 = nn.Conv2d(500, 500, 10, stride=1)
        self.pool4 = nn.MaxPool2d(2, stride=1, return_indices=True)
        self.encoding_layers = [self.conv1, self.pool1, self.conv2, self.pool2,
                                self.conv3, self.pool3, self.conv4, self.pool4]
        
        self.unpool4 = nn.MaxUnpool2d(2, stride=1)
        self.deconv4 = nn.ConvTranspose2d(500, 500, 10, stride=1)
        self.unpool3 = nn.MaxUnpool2d(2, stride=1)
        self.deconv3 = nn.ConvTranspose2d(500, 200, 10, stride=1)
        self.unpool2 = nn.MaxUnpool2d(2, stride=1)
        self.deconv2 = nn.ConvTranspose2d(200, 100, 10, stride=2)
        self.unpool1 = nn.MaxUnpool2d(2, stride=1)
        self.deconv1 = nn.ConvTranspose2d(100, 1, 10, stride=2)
        self.decoding_layers = [self.unpool4, self.deconv4, self.unpool3, self.deconv3,
                                self.unpool2, self.deconv2, self.unpool1, self.deconv1]
        
        self.to(device)
    
    def forward(self, x):
        encoded = []
        idxs = []
        for layer in self.encoding_layers:
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
                encoded.append(x)
            else:
                x, idx = layer(x)
                idxs.append(idx)
                x = F.leaky_relu(x)
        for i in range(len(self.decoding_layers)):
            layer = self.decoding_layers[i]
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
            else:
                x = layer(x, idxs.pop())
                if i != len(self.decoding_layers):
                    x = F.leaky_relu(x)
                
        return (encoded, x)        
    
    def infer(self, x):
        ret = []
        for layer in self.encoding_layers:
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
            else:
                x, _ = layer(x)
            ret.append(x)
            if isinstance(layer, nn.MaxPool2d):
                x = F.leaky_relu(x)
        return ret

    