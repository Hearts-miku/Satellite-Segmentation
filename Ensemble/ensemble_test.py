import gc
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from collections import OrderedDict
from torch.autograd import Variable
import torchvision
from dataset import MyTestData
from model import Deconv
import densenet
import numpy as np
from datetime import datetime
import os
import glob
import pdb
import argparse
from PIL import Image
from os.path import expanduser
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

home = expanduser("~")


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='../Data/val/')  # training dataset
parser.add_argument('--output_dir', default='./output')  # training dataset
parser.add_argument('--para_dir', default='parameters')  # training dataset
parser.add_argument('--b', type=int, default=1)  # batch size
parser.add_argument('--q', default='densenet121')  # save checkpoint parameters
opt = parser.parse_args()


def to_label(outputs):
    o = np.argmax(outputs, axis=1)
    return o


def main():
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)
    bsize = opt.b
    # models
    feature0 = UNet(num_classes=2)
    feature1 = FCN(num_classes=2)
    feature2 = FCN_ResNet(num_classes=2, backbone='resnet50', out_stride=32, mult_grid=False)

    feature0.cuda()
    feature0.eval()
    feature1.cuda()
    feature1.eval()
    feature2.cuda()
    feature2.eval()

    sb0 = torch.load('%s/' % (opt.para_dir) + 'UNet' + '_feature.pth')
    sb1 = torch.load('%s/' % (opt.para_dir) + 'FCNVGG' + '_feature.pth')
    sb2 = torch.load('%s/' % (opt.para_dir) + 'FCNResNet' + '_feature.pth')


    feature0.load_state_dict(sb0)
    feature1.load_state_dict(sb1)
    feature2.load_state_dict(sb2)

    loader = torch.utils.data.DataLoader(
        MyTestData(opt.input_dir),
        batch_size=bsize, shuffle=False, num_workers=0, pin_memory=True)
    for ib, (data, img_name, img_size) in enumerate(loader):
        out = torch.zeros((512, 512))
        inputs = Variable(data).cuda()


        # First get the output of all networks
        outputs0 = feature0(inputs)
        outputs0 = outputs0.data.cpu().squeeze(1).numpy()

        outputs1 = feature1(inputs)
        outputs1 = outputs1.data.cpu().squeeze(1).numpy()

        outputs2 = feature2(inputs)
        outputs2 = outputs2.data.cpu().squeeze(1).numpy()


        # Weighted voting
        '''
        outputs0 = to_label(outputs0)
        outputs1 = to_label(outputs1)
        outputs2 = to_label(outputs2)
        outputs = outputs0*0.4+outputs1*0.3+outputs2*0.3
        outputs[np.where(outputs > 0.5)] = 1
        outputs[np.where(outputs <= 0.5)] = 0
        '''

        # Average
        #outputs = np.mean(np.array([outputs0, outputs1, outputs2]), axis=0)


        # Weighted average
        outputs = np.mean(np.array([outputs0*0.7, outputs1*0.15, outputs2*0.15]), axis=0)


        # Median
        #outputs = np.median(np.array([outputs0, outputs1, outputs2]), axis=0)

        # If use voting method, please annotation this line
        outputs = to_label(outputs)

        for ii, msk in enumerate(outputs):
            msk = (msk * 255).astype(np.uint8)
            msk = Image.fromarray(msk)
            msk = msk.resize((img_size[0][ii], img_size[1][ii]))
            msk.save('%s/%s.png' % (opt.output_dir, img_name[ii]), 'PNG')
            #print(np.array(msk),gt[ib])
            #print(compute_f1(np.array(msk), gt[ib]*255))




if __name__ == "__main__":
    main()