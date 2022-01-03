import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
from pylab import plt
from tqdm import tqdm

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

'''
https://zhuanlan.zhihu.com/p/25071913
WGAN modified of DCGAN in:
1. remove sigmoid in the last layer of discriminator(classification -> regression)                                       
# 回归问题,而不是二分类概率
2. no log Loss (Wasserstein distance)
3. clip param norm to c (Wasserstein distance and Lipschitz continuity)
4. No momentum-based optimizer, use RMSProp，SGD instead

explanation of GAN：
collapse mode ->KL diverse
digit unstability-> comflict between KL Divergence and JS Divergence
'''
class Config:
    lr = 0.00005
    nz = 100 # noise dimension
    image_size = 64
    image_size2 = 64
    nc = 3 # chanel of img 
    ngf = 64 # generate channel
    ndf = 64 # discriminative channel
    beta1 = 0.5
    batch_size = 32
    max_epoch = 50 # =1 when debug
    workers = 2
    gpu = True # use gpu or not
    clamp_num=0.01# WGAN clip gradient

def weight_init(m):
        # weight_initialization: important for wgan
        class_name=m.__class__.__name__
        if class_name.find('Conv')!=-1:
            m.weight.data.normal_(0,0.02)
        elif class_name.find('Norm')!=-1:
            m.weight.data.normal_(1.0,0.02)
    #     else:print(class_name)

if __name__ == "__main__":
    opt=Config()

    # data preprocess
    transform=transforms.Compose([
                    transforms.Resize(opt.image_size) ,
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*3,[0.5]*3)
                    ])

    dataset=CIFAR10(root=r'D:\workstation\GitHub\DeepMindStudy\data\cifar10\data',transform=transform,download=True)
    # dataloader with multiprocessing
    dataloader=t.utils.data.DataLoader(dataset,
                                    opt.batch_size,
                                    shuffle = True,
                                    num_workers=opt.workers)

    # 搭建模型
    netg = nn.Sequential(
                nn.ConvTranspose2d(opt.nz,opt.ngf*8,4,1,0,bias=False),
                nn.BatchNorm2d(opt.ngf*8),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(opt.ngf*8,opt.ngf*4,4,2,1,bias=False),
                nn.BatchNorm2d(opt.ngf*4),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(opt.ngf*4,opt.ngf*2,4,2,1,bias=False),
                nn.BatchNorm2d(opt.ngf*2),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(opt.ngf*2,opt.ngf,4,2,1,bias=False),
                nn.BatchNorm2d(opt.ngf),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(opt.ngf,opt.nc,4,2,1,bias=False),
                nn.Tanh()
            )

    netd = nn.Sequential(
                nn.Conv2d(opt.nc,opt.ndf,4,2,1,bias=False),
                nn.LeakyReLU(0.2,inplace=True),
                
                nn.Conv2d(opt.ndf,opt.ndf*2,4,2,1,bias=False),
                nn.BatchNorm2d(opt.ndf*2),
                nn.LeakyReLU(0.2,inplace=True),
                
                nn.Conv2d(opt.ndf*2,opt.ndf*4,4,2,1,bias=False),
                nn.BatchNorm2d(opt.ndf*4),
                nn.LeakyReLU(0.2,inplace=True),
                
                nn.Conv2d(opt.ndf*4,opt.ndf*8,4,2,1,bias=False),
                nn.BatchNorm2d(opt.ndf*8),
                nn.LeakyReLU(0.2,inplace=True),
                
                nn.Conv2d(opt.ndf*8,1,4,1,0,bias=False),
                # Modification 1: remove sigmoid
                # nn.Sigmoid()
            )

    netd.apply(weight_init)
    netg.apply(weight_init)

    # modification 2: Use RMSprop instead of Adam
    # optimizer
    optimizerD = RMSprop(netd.parameters(),lr=opt.lr ) 
    optimizerG = RMSprop(netg.parameters(),lr=opt.lr )  

    # modification3: No Log in loss
    # criterion
    # criterion = nn.BCELoss()

    fix_noise = Variable(t.FloatTensor(opt.batch_size,opt.nz,1,1).normal_(0,1))
    if opt.gpu:
        fix_noise = fix_noise.cuda()
        netd.cuda()
        netg.cuda()

    print(netd)
    print(netg)

    # begin training
    print('begin training, be patient...')
    # one=t.FloatTensor([1])
    # one = t.tensor(1, dtype=t.float)
    one = t.rand(32, 1, 1, 1)
    mone=-1*one

    bar = tqdm(total = opt.max_epoch)
    for epoch in range(opt.max_epoch):
        bar.update(1)
        subbar = tqdm(total = len(dataloader), leave = False)
        for ii, data in enumerate(dataloader,0):
            subbar.update(1)
            real,_=data
            input = Variable(real)
            noise = t.randn(input.size(0),opt.nz,1,1)
            noise = Variable(noise)
            
            if opt.gpu:
                one = one.cuda()
                mone = mone.cuda()
                noise = noise.cuda()
                input = input.cuda()

            # modification: clip param for discriminator
            for parm in netd.parameters():
                    parm.data.clamp_(-opt.clamp_num,opt.clamp_num)
            
            # ----- train netd -----
            netd.zero_grad()
            ## train netd with real img
            output=netd(input)
            output.backward(one)
            ## train netd with fake img
            fake_pic=netg(noise).detach()
            output2=netd(fake_pic)
            output2.backward(mone)
            optimizerD.step()
            
            # ------ train netg -------
            # train netd more: because the better netd is,
            # the better netg will be
            if (ii+1)%5 ==0:
                netg.zero_grad()
                noise.data.normal_(0,1)
                fake_pic=netg(noise)
                output=netd(fake_pic)
                output.backward(one)
                optimizerG.step()
                if ii%100==0:pass
        subbar.close()
        fake_u=netg(fix_noise)
        imgs = make_grid(fake_u.data*0.5+0.5).cpu() # CHW
        # plt.imshow(imgs.permute(1,2,0).numpy()) # HWC
        # plt.show()
        plt.savefig('WGAN/output/' + str(epoch) + '-' + str(ii) + '.png')
    bar.close()

    t.save(netd.state_dict(),r'WGAN\modelepoch_wnetd.pth')
    t.save(netg.state_dict(),r'WGAN\modelepoch_wnetg.pth')

    netd.load_state_dict(t.load(r'WGAN\model\epoch_wnetd.pth'))
    netg.load_state_dict(t.load(r'WGAN\modelepoch_wnetg.pth'))

    noise = t.randn(64,opt.nz,1,1).cuda()
    noise = Variable(noise)
    fake_u=netg(noise)
    imgs = make_grid(fake_u.data*0.5+0.5).cpu() # CHW
    plt.figure(figsize=(5,5))
    # plt.imshow(imgs.permute(1,2,0).numpy()) # HWC
    # plt.show()
    plt.savefig(r'WGAN\output\lastresult.png')