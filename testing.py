import torch
from matplotlib.pyplot import imshow
from skimage.io import imread
import matplotlib.pyplot as plt
import input_processing

net = torch.load('C:\\Users\Justin Wang\Desktop\JustinLWang\intermDat\SCAE2\\tempnet', map_location='cpu')
test_img_dir = 'D:\\Data\\TIMIT\\786\\spectro.png'
test_img = torch.as_tensor(imread(test_img_dir) / 255)
test_img = input_processing.dynamic_one_to_one_process(test_img, 10, net.encoding_layers)
test_img = torch.reshape(test_img, [1,1,test_img.shape[0], test_img.shape[1]]).detach()

encoded, out_img = net(test_img.float())
out_img = out_img.detach()

mse = ((test_img.float() - out_img.float())**2).mean()

cmap = plt.cm.viridis
norm = plt.Normalize(vmin=test_img.min(), vmax=test_img.max())

out_img = torch.reshape(out_img, [out_img.shape[2], out_img.shape[3]])
test_img = torch.reshape(test_img, [test_img.shape[2], test_img.shape[3]])

test_img_out = cmap(norm(test_img))
out_img_out = cmap(norm(out_img))

# save the image
plt.imsave('C:\\Users\Justin Wang\Desktop\\paper\\outImg.png', out_img_out)
plt.imsave('C:\\Users\Justin Wang\Desktop\\paper\\testImg.png', test_img_out)

print(mse)


