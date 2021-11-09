import gc
import torch
import torchvision
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.autograd import Variable
from dataset import MyData
from model import Deconv
import densenet
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import pdb
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from models.UNet import UNet
from models.SegNet import SegNet
from models.FCN8s import FCN
from models.BiSeNet import BiSeNet
from models.BiSeNetV2 import BiSeNetV2
from models.PSPNet.pspnet import PSPNet
from models.DeeplabV3Plus import Deeplabv3plus_res50
from models.FCN_ResNet import FCN_ResNet
from models.DDRNet import DDRNet
from models.HRNet import HighResolutionNet

# Please change the batch size and epoch based on your hardware
batch_size = 1
epoch = 500

'''
Select method
Here provide 'DenseNet', 'UNet', 'FCNVGG', 'FCNResNet'
You can build other networks which in 'models' by yourself.
'''
method = 'UNet'

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='../Data/train/')  # training dataset
parser.add_argument('--val_dir', default='../Data/val/')  # training dataset
parser.add_argument('--check_dir', default='./parameters')  # save checkpoint parameters
parser.add_argument('--q', default='densenet121')  # save checkpoint parameters
parser.add_argument('--b', type=int, default=batch_size)  # batch size
parser.add_argument('--e', type=int, default=epoch)  # epoches
opt = parser.parse_args()


def to_label(outputs):
    o = np.argmax(outputs, axis=1)
    return o

#save for densenet
def save(check_dir, feature, deconv):
    filename = ('%s/deconv.pth' % (check_dir))
    torch.save(deconv.state_dict(), filename)
    filename = ('%s/dense_feature.pth' % (check_dir))
    torch.save(feature.state_dict(), filename)

# dave for other networks
def save1(check_dir, feature, method):
    filename = ('%s/'% (check_dir) + method + '_feature.pth')
    torch.save(feature.state_dict(), filename)


def make_image_grid(img, mean, std):
    img = make_grid(img)
    for i in range(3):
        img[i] *= std[i]
        img[i] += mean[i]
    return img

# change into one-hot code
def one_hot(lbl):
    tmp = lbl.cpu().numpy()
    batch = lbl.shape[0]
    onehot = torch.zeros((batch, 2, 512, 512))

    for b in range(batch):
        onehot[b, 0, np.where(tmp[b, 0] == 0)[0], np.where(tmp[b, 0] == 0)[1]] = 1
        onehot[b, 1, np.where(tmp[b, 0] == 1)[0], np.where(tmp[b, 0] == 1)[1]] = 1

    onehot = onehot.to('cuda')
    return onehot

def main():
    # tensorboard writer
    '''
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.system('rm -rf ./runs/*')
    #writer = SummaryWriter('./runs/exp')
    if not os.path.exists('./runs'):
        os.mkdir('./runs')
    std = [.229, .224, .225]
    mean = [.485, .456, .406]
    '''

    train_dir = opt.train_dir
    val_dir = opt.val_dir
    check_dir = opt.check_dir


    bsize = opt.b
    iter_num = opt.e  # training iterations
    train_loss = np.zeros((int(60/bsize), iter_num))

    if not os.path.exists(check_dir):
        os.mkdir(check_dir)

    # models
    use_Densenet = False
    # If use densenet
    if method == 'DenseNet':
        use_Densenet = True
        feature = getattr(densenet, opt.q)(pretrained=False)

    # If use other networks, see 'models' folder
    elif method == 'UNet':
        feature = UNet(num_classes=2)
    elif method == 'FCNVGG':
        feature = FCN(num_classes=2)
    elif method == 'FCNResNet':
        feature = FCN_ResNet(num_classes=2, backbone='resnet50', out_stride=32, mult_grid=False)


    feature.cuda()
    deconv = Deconv(opt.q)
    deconv.cuda()

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        MyData(train_dir, transform=True, crop=False, hflip=False, vflip=False),
        batch_size=bsize, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        MyData(val_dir,  transform=True, crop=False, hflip=False, vflip=False),
        batch_size=bsize, shuffle=True, num_workers=0, pin_memory=True)

    optimizer = torch.optim.Adam([
        {'params': feature.parameters(), 'lr': 1e-4},
        {'params': deconv.parameters(), 'lr': 1e-3},
        ])

    for it in range(iter_num):
        if use_Densenet:
            for ib, (data, lbl) in enumerate(train_loader):
                inputs = Variable(data).cuda()
                lbl = Variable(lbl.float().unsqueeze(1)).cuda()
                feats = feature(inputs)

                msk = deconv(feats)
                loss = F.binary_cross_entropy_with_logits(msk, lbl)
                deconv.zero_grad()
                feature.zero_grad()
                loss.backward()
                optimizer.step()
                print('loss: %.4f (epoch: %d, step: %d)' % (loss.item(), it, ib))
                train_loss[ib, it]=loss.item()
                del inputs, msk, lbl, loss, feats
                gc.collect()

            save(check_dir, feature, deconv)


        else:
            for ib, (data, lbl) in enumerate(train_loader):
                inputs = Variable(data).cuda()
                lbl = Variable(lbl.float().unsqueeze(1)).cuda()
                feats = feature(inputs)
                msk = feats
                lbl = one_hot(lbl)
                loss = F.binary_cross_entropy_with_logits(msk, lbl)

                deconv.zero_grad()
                feature.zero_grad()
                loss.backward()

                optimizer.step()
                print('loss: %.4f (epoch: %d, step: %d)' % (loss.item(), it, ib))
                train_loss[ib, it] = loss.item()
                del inputs, msk, lbl, loss, feats
                gc.collect()

            save1(check_dir, feature, method)

    # Output training loss
    for i in range(train_loss.shape[0]):
        path = 'loss/' + str(i) + '.jpg'
        plt.figure()
        plt.plot(train_loss[i, :])
        plt.savefig(path)
        plt.close()


if __name__ == "__main__":
    main()

