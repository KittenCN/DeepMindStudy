import os
import matplotlib.pyplot as plt
from datetime import date,datetime
import logging
from tqdm import tqdm
import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim
from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 96
img_rate = 3  # 64:1, 128:5
base_channels = img_size
g_file = r'./wgan-gp-model/G.ckpt'
d_file = r'./wgan-gp-model/D.ckpt'
img_folder = r'./datasets/face'
sample_dir = r'./wgan-gp-results'
writer = SummaryWriter(r'/root/tf-logs')
log_file = r'./log/logger.log'
D_cicle = 5

# --------
# 定义网络
# --------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=base_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchN1 = nn.InstanceNorm2d(base_channels, affine=True)
        self.LeakyReLU1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchN2 = nn.InstanceNorm2d(base_channels*2, affine=True)
        self.LeakyReLU2 = nn.LeakyReLU(0.2, inplace=True)       

        self.conv3 = nn.Conv2d(in_channels=base_channels*2, out_channels=base_channels*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchN3 = nn.InstanceNorm2d(base_channels*4, affine=True)
        self.LeakyReLU3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv2d(in_channels=base_channels*4, out_channels=base_channels*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchN4 = nn.InstanceNorm2d(base_channels*8, affine=True)
        self.LeakyReLU4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv5 = nn.Conv2d(in_channels=base_channels*8, out_channels=1, kernel_size=4, bias=False)
#         self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.LeakyReLU1(self.batchN1(self.conv1(x)))
        x = self.LeakyReLU2(self.batchN2(self.conv2(x)))
        x = self.LeakyReLU3(self.batchN3(self.conv3(x)))
        x = self.LeakyReLU4(self.batchN4(self.conv4(x)))
        x = self.conv5(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ConvT1 = nn.ConvTranspose2d(in_channels=100, out_channels=base_channels*8, kernel_size=4, bias=False) # 这里的in_channels是和初始的随机数有关
        self.batchN1 = nn.BatchNorm2d(base_channels*8)
        self.relu1 = nn.ReLU()
        
        self.ConvT2 = nn.ConvTranspose2d(in_channels=base_channels*8, out_channels=base_channels*4, kernel_size=4, stride=2, padding=1, bias=False) # 这里的in_channels是和初始的随机数有关
        self.batchN2 = nn.BatchNorm2d(base_channels*4)
        self.relu2 = nn.ReLU()        
        
        self.ConvT3= nn.ConvTranspose2d(in_channels=base_channels*4, out_channels=base_channels*2, kernel_size=4, stride=2, padding=1, bias=False) # 这里的in_channels是和初始的随机数有关
        self.batchN3 = nn.BatchNorm2d(base_channels*2)
        self.relu3 = nn.ReLU()

        self.ConvT4 = nn.ConvTranspose2d(in_channels=base_channels*2, out_channels=base_channels, kernel_size=4, stride=2, padding=1, bias=False) # 这里的in_channels是和初始的随机数有关
        self.batchN4 = nn.BatchNorm2d(base_channels)
        self.relu4 = nn.ReLU()
        
        self.ConvT5 = nn.ConvTranspose2d(in_channels=base_channels, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh() # 激活函数
        
    def forward(self, x):
        x = self.relu1(self.batchN1(self.ConvT1(x)))
        x = self.relu2(self.batchN2(self.ConvT2(x)))
        x = self.relu3(self.batchN3(self.ConvT3(x)))
        x = self.relu4(self.batchN4(self.ConvT4(x)))
        x = self.ConvT5(x)
        x = self.tanh(x)
        return x

# 定义辅助函数
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

if __name__ == "__main__":
    # 将日志保存到文件
    logging.basicConfig(filename=log_file,level=logging.INFO)
    # ----------
    # 加载数据集
    # ----------
    trans = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(img_folder, transform=trans) # 数据路径
    dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=450, # 批量大小
                                        shuffle=True, # 乱序
                                        num_workers=6 # 多进程
                                        )
    # ----------
    # 初始化网络
    # ----------
    D = Discriminator().to(device) # 定义分类器
    G = Generator().to(device) # 定义生成器
    print(D)
    print(G)
    if os.path.exists(g_file) and os.path.exists(d_file):
        G.load_state_dict(torch.load(g_file))
        D.load_state_dict(torch.load(d_file))
        G.eval()
        D.eval()
        print("load old model")
    else:
        print("new model")
    
    # -----------------------
    # 定义损失函数和优化器
    # -----------------------
    learning_rate = 0.0002
    d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    # 每3次降低学习率
    d_exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=1000, gamma=0.95)
    g_exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=1000, gamma=0.95)
    # 定义惩罚系数
    penalty_lambda = 0.1
    # --------
    # 开始训练
    # --------
    num_epochs = 10000
    test_noise = Variable(torch.FloatTensor(40, 100, img_rate, img_rate).normal_(0, 1)).to(device) # 用于测试绘图
    total_step = len(dataloader) # 依次epoch的步骤
    # ------------------
    # 一开始学习率快一些
    # ------------------
    bar = tqdm(total=num_epochs, leave=False)
    for epoch in range(num_epochs):
        bar.set_description("epochs:")
        bar.update(1)
        subbar = tqdm(total=len(dataloader), leave=False)
        for i, (images, _) in enumerate(dataloader):
            subbar.set_description("ite:")
            subbar.update(1)
            batch_size = images.size(0)
            images = images.reshape(batch_size, 3, img_size, img_size).to(device)
            # 创造real label和fake label
            real_labels = torch.ones(batch_size, 1).to(device) # real的pic的label都是1
            fake_labels = torch.zeros(batch_size, 1).to(device) # fake的pic的label都是0
            noise = Variable(torch.randn(batch_size, 100, img_rate, img_rate)).to(device) # 随机噪声，生成器输入
            exsubbar = tqdm(total=D_cicle, leave=False)
            for j in range(D_cicle):
                # ---------------------
                # 开始训练discriminator
                # ---------------------
                exsubbar.set_description("D_cicle:")
                exsubbar.update(1)
                D.train()
                G.train()
                # 首先计算真实的图片
                outputs = D(images)
                d_loss_real = -torch.mean(outputs)

                # 接着使用生成器训练得到图片, 放入判别器
                fake_images = G(noise)
                outputs = D(fake_images)
                d_loss_fake = torch.mean(outputs)

                # 生成gradient penalty

                # 1. P_data与P_G的中间区域
                alpha = torch.rand((batch_size, 1, 1, 1)).to(device)
                # print(images.size())
                # print(fake_images.size())
                x_hat = alpha * images.data + (1 - alpha) * fake_images.data
                x_hat.requires_grad = True

                # 2. 计算penalty region处的梯度
                pred_hat = D(x_hat)
                gradient = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(device), create_graph=True, retain_graph=True) # 计算梯度
                gradient_penalty = penalty_lambda * ((gradient[0].view(gradient[0].size()[0], -1).norm(p=2,dim=1)-1)**2).mean()
                Wasserstein_D = d_loss_real + d_loss_fake

                # 三个loss相加, 反向传播进行优化
                d_loss = d_loss_real + d_loss_fake + gradient_penalty
                g_optimizer.zero_grad() # 两个优化器梯度都要清0
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()   
            exsubbar.close()
            normal_noise = Variable(torch.randn(batch_size, 100, img_rate, img_rate)).normal_(0, 1).to(device)
            fake_images = G(normal_noise) # 生成假的图片
            outputs = D(fake_images) # 放入辨别器
            g_loss = -torch.mean(outputs) # 希望生成器生成的图片判别器可以判别为真
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            # ----------
            # 打印结果
            # ---------
            if (i+1) % 1 == 0:
#                 t = datetime.now() #获取现在的时间
#                 logging.info('Time {}, Epoch [{}/{}], Step [{}/{}], d_loss_real:{:.4f} + d_loss_fake:{:.4f} + gradient_penalty:{:.4f} = d_loss: {:.4f}, g_loss: {:.4f}, d_lr={:.6f},g_lr={:.6f}'
#                     .format(t, epoch, num_epochs, i+1, total_step, d_loss_real.item(), d_loss_fake.item(), gradient_penalty.item(), d_loss.item(), g_loss.item(),
#                             d_optimizer.param_groups[0]['lr'], g_optimizer.param_groups[0]['lr']))
                writer.add_scalar('lossd/d_loss_real', d_loss_real.item(), i)
                writer.add_scalar('lossd/d_loss_fake', d_loss_fake.item(), i)
                writer.add_scalar('lossd/d_loss', d_loss.item(), i)
                writer.add_scalar('lossg/g_loss', g_loss.item(), i)
                writer.add_scalar('loss/gradient_penalty', gradient_penalty.item(), i)
                writer.add_scalar('loss/Wasserstein_D', Wasserstein_D.item(), i)
        subbar.close()
        d_exp_lr_scheduler.step()
        g_exp_lr_scheduler.step()    
        
        # ----------
        # 打印结果
        # ---------
        if (epoch+1) % 1 == 0:
            t = datetime.now() #获取现在的时间
            logging.info('Time {}, Epoch [{}/{}], Step [{}/{}], d_loss_real:{:.4f} + d_loss_fake:{:.4f} + gradient_penalty:{:.4f} = d_loss: {:.4f}, g_loss: {:.4f}, d_lr={:.6f},g_lr={:.6f}'
                .format(t, epoch, num_epochs, i+1, total_step, d_loss_real.item(), d_loss_fake.item(), gradient_penalty.item(), d_loss.item(), g_loss.item(),
                        d_optimizer.param_groups[0]['lr'], g_optimizer.param_groups[0]['lr']))
            
        # -----------
        # 结果的保存
        # ----------
        # 每一个epoch显示图片(这里切换为eval模式)
        if (epoch + 1) % 1 == 0:
            G.eval()
            test_images = G(test_noise)
#             writer.add_image('fake_images-norm-{}.png'.format(epoch+1), test_images, 1, dataformats='HWC')
            save_image(denorm(test_images), os.path.join(sample_dir, 'f_images-{}.png'.format(epoch+1)))
            # Save the model checkpoints 
            torch.save(G.state_dict(), g_file)
            torch.save(D.state_dict(), d_file)
    bar.close()