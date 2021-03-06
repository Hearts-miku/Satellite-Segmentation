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
home = expanduser("~")

'''
Only used for testing DenseNet
'''


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='../Data/val/')  # training dataset
parser.add_argument('--output_dir', default='./output')  # training dataset
parser.add_argument('--para_dir', default='parameters')  # training dataset
parser.add_argument('--b', type=int, default=1)  # batch size
parser.add_argument('--q', default='densenet121')  # save checkpoint parameters
opt = parser.parse_args()
print(opt)


def main():
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)
    bsize = opt.b
    # models
    feature = getattr(densenet, opt.q)(pretrained=False)
    feature.cuda()
    feature.eval()
    sb = torch.load('%s/dense_feature.pth'%opt.para_dir)
    feature.load_state_dict(sb)

    deconv = Deconv(opt.q)
    deconv.cuda()
    deconv.eval()
    sb = torch.load('%s/deconv.pth' % opt.para_dir)

    deconv.load_state_dict(sb)
    loader = torch.utils.data.DataLoader(
        MyTestData(opt.input_dir),
        batch_size=bsize, shuffle=True, num_workers=0, pin_memory=True)
    for ib, (data, img_name, img_size) in enumerate(loader):
        inputs = Variable(data).cuda()
        feats = feature(inputs)
        outputs = deconv(feats)
        outputs = F.sigmoid(outputs)

        outputs = outputs.data.cpu().squeeze(1).numpy()
        for ii, msk in enumerate(outputs):
            msk[np.where(msk>=0.5)]=1
            msk[np.where(msk < 0.5)] = 0
            msk = (msk * 255).astype(np.uint8)
            msk = Image.fromarray(msk)
            msk = msk.resize((img_size[0][ii], img_size[1][ii]))
            msk.save('%s/%s.png' % (opt.output_dir, img_name[ii]), 'PNG')


if __name__ == "__main__":
    main()

