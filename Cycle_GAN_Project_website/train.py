import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *
from util import *

import itertools
import matplotlib.pyplot as plt

from torchvision import transforms

MEAN = 0.5
STD = 0.5

NUM_WORKER = 0

def test():
    ## 트레이닝 파라메터 설정하기
    mode = "test"
    train_continue ="on"

    lr = "on"


    data_dir = "./static/img"
    ckpt_dir = "./pth"
    result_dir = "./static/result"

    task = "cyclegan"
    opts = ['direction', np.asarray(0).astype(np.float)]

    ny = 256
    nx = 256
    nch = 3
    nker = 64

 
    norm = 'inorm'

    network = "CycleGAN"
    learning_type = "plain"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    ## 디렉토리 생성하기
    result_dir_test = os.path.join(result_dir, 'test')

        # os.makedirs(os.path.join(result_dir_test, 'numpy'))

    ## 네트워크 학습하기
    if mode == 'test':
        transform_test = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=MEAN, std=STD)])

        dataset_test_a = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_test, task=task,
                                 data_type='a')
        loader_test_a = DataLoader(dataset_test_a, batch_size=2, shuffle=False, num_workers=NUM_WORKER)



    ## 네트워크 생성하기
    if network == "CycleGAN":
        netG_a2b = CycleGAN(in_channels=nch, out_channels=nch, nker=nker, norm=norm, nblk=9).to(device)

        init_weights(netG_a2b, init_type='normal', init_gain=0.02)
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x: (x * STD) + MEAN

    # TRAIN MODE
    if mode == "test":
        netG_a2b= load(ckpt_dir=ckpt_dir,netG_a2b=netG_a2b)

        with torch.no_grad():
            netG_a2b.eval()

            for batch, data in enumerate(loader_test_a, 1):
                # forward pass
                input_a = data['data_a'].to(device)
                output_b = netG_a2b(input_a)
                output_b = fn_tonumpy(fn_denorm(output_b))

                for j in range(input_a.shape[0]):

                    output_b_ = output_b[j]
                    output_b_ = np.clip(output_b_, a_min=0, a_max=1)
                    plt.imsave(os.path.join(result_dir_test, 'out3.png'), output_b_)
                    print("TEST 완료")
