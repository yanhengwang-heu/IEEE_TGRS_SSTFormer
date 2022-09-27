from scipy.io import loadmat
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


pred = Image.open('barbara.tif')
data_label = loadmat("E:/IEEE_TGRS_SSTlFormer-main/data/Barbara/barbara_gtChanges.mat")['HypeRvieW']

# pred = Image.open('BayArea.tif')
# data_label = loadmat("E:/IEEE_TGRS_SSTlFormer-main/data/BayArea/bayArea_gtChanges2.mat")['HypeRvieW']

# pred = Image.open('barbara.tif')
# data_label = loadmat("E:/IEEE_TGRS_SSTlFormer-main/data/BayArea/bayArea_gtChanges2.mat")['HypeRvieW']

data_label[data_label!=0]=1

pred = np.array(pred)
pred = pred/255
output=pred*data_label
output=output+data_label
# plt.subplot(1,1,1)
# plt.imshow(output, colors.ListedColormap(color_matrix))
# # plt.show()

im=Image.fromarray(output/2*255)
im=im.convert('RGB')
im.show()
