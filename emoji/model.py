import torch
import torch.nn as nn

# 通过最近邻插值的方式将长宽加倍
def amplify_img(imgs):
    return nn.functional.interpolate(imgs, torch.Size([imgs.shape[-2] * 2, imgs.shape[-1] * 2]), mode='nearest')

# 生成器
class G_net(nn.Module):
    def __init__(self):
        super(G_net, self).__init__()
        self.expand = nn.Sequential(
            nn.Linear(128, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convTran1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convTran2 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convTran3 = nn.Sequential(
            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convTran4 = nn.Sequential(
            nn.ConvTranspose2d(512, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convTran5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convTran6 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1, stride=1, bias=False),
            # 将输出约束到[-1,1]
            nn.Tanh()
        )

    def forward(self, img_seeds):
        img_seeds = self.expand(img_seeds)
        # 将线性数据重组为二维图片
        imgs = img_seeds.view(-1, 64, 8, 8)
        # 用转置卷积放大图片
        imgs = self.convTran1(imgs)
        # 用最近邻插值放大图片
        imgs = amplify_img(imgs)
        # 压缩图片为16x16
        imgs = self.conv1(imgs)
        # 用转置卷积放大图片
        imgs = self.convTran2(imgs)
        # 用最近邻插值放大图片
        imgs = amplify_img(imgs)
        # 压缩图片为32x32
        imgs = self.conv2(imgs)
        # 用转置卷积放大图片
        imgs = self.convTran3(imgs)
        # 用最近邻插值放大图片
        imgs = amplify_img(imgs)
        # 压缩图片为64x64
        imgs = self.conv3(imgs)
        # 用转置卷积放大图片
        imgs = self.convTran4(imgs)
        # 用最近邻插值放大图片
        imgs = amplify_img(imgs)
        # 压缩图片为128x128
        imgs = self.conv4(imgs)
        # 用转置卷积放大图片
        imgs = self.convTran5(imgs)
        # 用最近邻插值放大图片
        imgs = amplify_img(imgs)
        # 压缩图片为256x256
        imgs = self.conv5(imgs)
        # 用转置卷积放大图片
        imgs = self.convTran6(imgs)
        # 用最近邻插值放大图片
        imgs = amplify_img(imgs)
        # 压缩图片为512x512
        imgs = self.conv6(imgs)
        # 1x1维度整合卷积层，整合为3通道图片
        imgs = self.conv7(imgs)
        return imgs

# 全局判别器，传统gan
class D_net_global(nn.Module):
    def __init__(self):
        super(D_net_global,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=6, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(5184, 1),
            #nn.Sigmoid(),
        )

    def forward(self, img):
        features = self.features(img)
        features = features.view(features.shape[0], -1)
        output = self.classifier(features)
        return output

# 局部判别器，patchgan
class D_net_patch(nn.Module):
    def __init__(self):
        super(D_net_patch,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, bias=False),
            #nn.Sigmoid(),
        )

    def forward(self, img):
        # 利用patch判别器输出矩阵
        features = self.features(img)
        # 全局平均池化,输出尺寸1x1
        features = nn.functional.adaptive_avg_pool2d(features, 1)
        # 展平
        features = features.view(features.shape[0], -1)
        #print("patch shape", features.shape)
        return features

# 返回对应的生成器
def get_G_model(from_old_model, device, model_path):
    model = G_net()
    # 从磁盘加载之前保存的模型参数
    if from_old_model:
        model.load_state_dict(torch.load(model_path))
    # 将模型加载到用于运算的设备的内存
    model = model.to(device)
    return model

# 返回全局判别器的模型
def get_D_model_global(from_old_model, device, model_path):
    model = D_net_global()
    # 从磁盘加载之前保存的模型参数
    if from_old_model:
        model.load_state_dict(torch.load(model_path))
    # 将模型加载到用于运算的设备的内存
    model = model.to(device)
    return model

# 返回局部判别器的模型
def get_D_model_patch(from_old_model, device, model_path):
    model = D_net_patch()
    # 从磁盘加载之前保存的模型参数
    if from_old_model:
        model.load_state_dict(torch.load(model_path))
    # 将模型加载到用于运算的设备的内存
    model = model.to(device)
    return model