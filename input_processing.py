import torch
import os
from skimage.io import imread
from torch.utils.data import Dataset


#Loader for spectrograms
class raw_spectro_loader(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len([name for name in os.listdir(self.root_dir)]) - 1
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(idx + 1), 'spectro.png')
        image = torch.as_tensor(imread(img_name) / 255)
        if self.transform:
            image = self.transform(image)
        return image

#Dynamic padding for 1-1 encoding/decoding
def dynamic_one_to_one_process(x, kernel_size, layers):
    rem = kernel_size%2
    procShape = list(x.shape)
    #process dimensions
    for dim in range(2):
        curSize = x.shape[dim]
        counter = 0
        for i in range(len(layers)):
            layerStride = layers[i].stride
            if not isinstance(layerStride, int):
                layerStride = layerStride[0]
            if curSize % 2 != rem and layerStride == 2:
                procShape[dim] += 2**counter
                curSize += 1
            if layerStride == 2:
                counter += 1
            curSize = (curSize - kernel_size)/layerStride + 1
    procX = torch.zeros(procShape)
    procX[:x.shape[0], :x.shape[1]] = x
    return procX 