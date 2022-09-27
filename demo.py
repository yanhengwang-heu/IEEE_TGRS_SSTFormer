import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
from sstvit import SSTViT
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import os

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['BayArea', 'Barbara', 'farmland','river'], default='farmland', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=90, help='number of evaluation')
parser.add_argument('--patches', type=int, default=5, help='number of patches')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=400, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--train_number', type=int, default=500, help='train_number')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
#-------------------------------------------------------------------------------
# 定位训练和测试样本
def chooose_train_and_test_point(train_data, test_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    # number_true = []
    # pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    return total_pos_train, total_pos_test, number_train, number_test
#-------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------
# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    # x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("**************************************************")

    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band
#-------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, num_classes):
    y_train = []
    y_test = []
    # y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("**************************************************")
    return y_train, y_test
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(train_loader):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()
        optimizer.zero_grad()
        batch_pred = model(batch_data_t1,batch_data_t2)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data_t1.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
#-------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(valid_loader):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data_t1,batch_data_t2)

        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data_t1.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre

def test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(test_loader):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data_t1,batch_data_t2)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre
#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
# prepare data
if args.dataset == 'BayArea':
    data_t1 = loadmat("./data/BayArea/Bay_Area_2013.mat")['HypeRvieW']
    data_t2 = loadmat("./data/BayArea/Bay_Area_2015.mat")['HypeRvieW']
    data_label = loadmat("./data/BayArea/bayArea_gtChanges2.mat")['HypeRvieW']
    uc_position = np.array(np.where(data_label==2)).transpose(1,0)
    c_position = np.array(np.where(data_label==1)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))

elif args.dataset == 'Barbara':
    data_t1 = loadmat("./data/Barbara/barbara_2013.mat")['HypeRvieW']
    data_t2 = loadmat("./data/Barbara/barbara_2014.mat")['HypeRvieW']
    data_label = loadmat("./data/Barbara/barbara_gtChanges.mat")['HypeRvieW']
    uc_position = np.array(np.where(data_label==2)).transpose(1,0)
    c_position = np.array(np.where(data_label==1)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))

elif args.dataset == 'river':
    data_t1 = loadmat('./data/river_dataset/river_before.mat')['river_before']
    data_t2 = loadmat('./data/river_dataset/river_after.mat')['river_after']
    data_label = loadmat('./data/river_dataset/groundtruth.mat')['lakelabel_v1']
    uc_position = np.array(np.where(data_label==0)).transpose(1,0)
    c_position = np.array(np.where(data_label==255)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))
    data_label = (data_label-data_label.min())/(data_label.max()-data_label.min())
    data_label[data_label==0]=2
elif args.dataset == 'farmland':
    data_t1 = loadmat('./data/farmland/China_Change_Dataset.mat')['T1']
    data_t2 = loadmat('./data/farmland/China_Change_Dataset.mat')['T2']
    data_label = loadmat('./data/farmland/China_Change_Dataset.mat')['Binary']
    uc_position = np.array(np.where(data_label==0)).transpose(1,0)
    c_position = np.array(np.where(data_label==1)).transpose(1,0)
    print((uc_position.shape[0],c_position.shape[0]))
    data_label[data_label==0]=2
else:
    raise ValueError("Unkknow dataset")

selected_uc = np.random.choice(uc_position.shape[0], int(args.train_number), replace = False)
selected_c = np.random.choice(c_position.shape[0], int(args.train_number), replace = False)
selected_uc_position=uc_position[selected_uc]
selected_c_position=c_position[selected_c]
TR = np.zeros(data_label.shape)
for i in range (int(args.train_number)):
    TR[selected_c_position[i][0],selected_c_position[i][1]]=1
    TR[selected_uc_position[i][0],selected_uc_position[i][1]]=2
#--------------测试样本-----------------
TE=data_label-TR

# color_mat = loadmat('./data/AVIRIS_colormap.mat')


num_classes = np.max(TR)
num_classes=int(num_classes)
# color_mat_list = list(color_mat)
# color_matrix = color_mat[color_mat_list[3]] #(17,3)
# normalize data by band norm
input1_normalize = np.zeros(data_t1.shape)
input2_normalize = np.zeros(data_t1.shape)
for i in range(data_t1.shape[2]):
    input_max = max(np.max(data_t1[:,:,i]),np.max(data_t2[:,:,i]))
    input_min = min(np.min(data_t1[:,:,i]),np.min(data_t2[:,:,i]))
    input1_normalize[:,:,i] = (data_t1[:,:,i]-input_min)/(input_max-input_min)
    input2_normalize[:,:,i] = (data_t2[:,:,i]-input_min)/(input_max-input_min)

height, width, band = data_t1.shape
print("height={0},width={1},band={2}".format(height, width, band))
#-------------------------------------------------------------------------------

if args.flag_test=='train':
    total_pos_train, total_pos_test, number_train, number_test = chooose_train_and_test_point(TR, TE, num_classes)
    mirror_image_t1 = mirror_hsi(height, width, band, input1_normalize, patch=args.patches)
    mirror_image_t2 = mirror_hsi(height, width, band, input2_normalize, patch=args.patches)
    x_train_band_t1, x_test_band_t1 = train_and_test_data(mirror_image_t1, band, total_pos_train, total_pos_test, patch=args.patches, band_patch=args.band_patches)
    x_train_band_t2, x_test_band_t2 = train_and_test_data(mirror_image_t2, band, total_pos_train, total_pos_test, patch=args.patches, band_patch=args.band_patches)
    y_train, y_test = train_and_test_label(number_train, number_test, num_classes)
    #-------------------------------------------------------------------------------
    # load data
    x_train_t1=torch.from_numpy(x_train_band_t1.transpose(0,2,1)).type(torch.FloatTensor) #[695, 200, 7, 7]
    x_train_t2=torch.from_numpy(x_train_band_t2.transpose(0,2,1)).type(torch.FloatTensor) #[695, 200, 7, 7]
    y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
    Label_train=Data.TensorDataset(x_train_t1,x_train_t2,y_train)
    x_test_t1=torch.from_numpy(x_test_band_t1.transpose(0,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
    x_test_t2=torch.from_numpy(x_test_band_t2.transpose(0,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
    y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
    Label_test=Data.TensorDataset(x_test_t1,x_test_t2,y_test)


    label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
    label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)

elif args.flag_test=='test':
    mirror_image_t1 = mirror_hsi(height, width, band, input1_normalize, patch=args.patches)
    mirror_image_t2 = mirror_hsi(height, width, band, input2_normalize, patch=args.patches)
    x1_true = np.zeros((height*width, args.patches, args.patches, band), dtype=float)
    x2_true = np.zeros((height*width, args.patches, args.patches, band), dtype=float)
    y_true=[]
    for i in range(height):
        for j in range(width):
            x1_true[i*width+j,:,:,:]=mirror_image_t1[i:(i+args.patches),j:(j+args.patches),:]
            x2_true[i*width+j,:,:,:]=mirror_image_t2[i:(i+args.patches),j:(j+args.patches),:]
            y_true.append(i)
    y_true = np.array(y_true)
    x1_true_band = gain_neighborhood_band(x1_true, band, args.band_patches, args.patches)
    x2_true_band = gain_neighborhood_band(x2_true, band, args.band_patches, args.patches)
    x1_true_band=torch.from_numpy(x1_true_band.transpose(0,2,1)).type(torch.FloatTensor)
    x2_true_band=torch.from_numpy(x2_true_band.transpose(0,2,1)).type(torch.FloatTensor)
    y_true=torch.from_numpy(y_true).type(torch.LongTensor)
    Label_true=Data.TensorDataset(x1_true_band,x2_true_band,y_true)
    label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)
    print('------测试数据加载完毕------')
#-------------------------------------------------------------------------------
# create model
model = SSTViT(
    image_size = args.patches,
    near_band = args.band_patches,
    num_patches = band,
    num_classes = num_classes,
    dim = 32,
    depth = 2,
    heads = 4,
    dim_head=16,
    mlp_dim = 8,
    b_dim = 512,
    b_depth = 3,
    b_heads = 8,
    b_dim_head= 32,
    b_mlp_head = 8,
    dropout = 0.2,
    emb_dropout = 0.1,
)

model = model.cuda()
# criterion
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//20, gamma=args.gamma)
#-------------------------------------------------------------------------------
if args.flag_test == 'test':
    if args.mode == 'ViT':
        model.load_state_dict(torch.load('./log/ssvit_v2_farmland.pth'))
    elif (args.mode == 'CAF') & (args.patches == 1):
        model.load_state_dict(torch.load('./SpectralFormer_pixel.pt'))
    elif (args.mode == 'CAF') & (args.patches == 7):
        model.load_state_dict(torch.load('./SpectralFormer_patch.pt'))
    else:
        raise ValueError("Wrong Parameters")
    model.eval()
    # output classification maps
    pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
    prediction_matrix = np.zeros((height, width), dtype=float)
    for i in range(height):
        for j in range(width):
            prediction_matrix[i,j] = pre_u[i*width+j] + 1
    plt.subplot(1,1,1)
    plt.imshow(prediction_matrix, colors.ListedColormap(color_matrix))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    from PIL import Image
    im=Image.fromarray((1-(prediction_matrix-1))*255)
    import ipdb; ipdb.set_trace()
    savemat('matrix.mat',{'P':prediction_matrix, 'label':label})
elif args.flag_test == 'train':
    print("start training")
    tic = time.time()
    for epoch in range(args.epoches):
        scheduler.step()

        # train model
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                        .format(epoch+1, train_obj, train_acc))

        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
            model.eval()
            tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    print("**************************************************")

print("Final result:")
print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
print(AA2)
print("**************************************************")
print("Parameter:")
torch.save(model.state_dict(), "log/ssvit_v2_farmland.pth")

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

print_args(vars(args))
