# 包加载与设备选择 ###########################################

from __future__ import print_function    # 从python未来的版本中import输出函数，主要是Python的print不需要括号，而Python3需要括号

import torch                        # torch中定义了多维张量的运算API，例如创建、索引、切片、连接、转置、加减乘除
import torch.nn as nn               # 包含搭建网络层的模块（Modules）和一系列的loss函数，例如全连接、卷积、池化、BN批处理、dropout、CrossEntropyLoss、MSELoss等
import torch.nn.functional as F     # 常用的激活函数relu、leaky_relu、sigmoid等
import torch.optim as optim         # 各种参数优化方法，例如SGD、AdaGrad、RMSProp、Adam等

from PIL import Image               # Python Imaging Library，是Python平台事实上的图像处理标准库
import matplotlib.pyplot as plt     # matplotlib是python常用的可视化库，提供一套与MATLAB相似的画图API

import torchvision.transforms as transforms    # 对PIL图片转换为Tensor，并且进行相关的转换，例如裁剪，缩放、归一化
import torchvision.models as models            # 常用模型，例如AlextNet、VGG、ResNet、DenseNet等，可以加载预训练或者没有预训练的模型

import copy                          # python中的拷贝包，用于拷贝模型的参数。  模型参数一般用字典格式保存，是Python的可变变量，需要deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 判断是否有GPU平台，如果有就使用GPU计算图片

# 图片加载与转换 ###########################################
# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128     # 如果有GPU就使用size为512的图片，否则使用128的，以减少在CPU上的计算负担

loader = transforms.Compose([        # transforms.Compose将多个transform的操作合并在一起
    transforms.Resize(imsize),       # 图像裁剪
    transforms.ToTensor()])          # 将pil图像转换为tensor，操作以后图像的数值范围是0-1，而不是0-255


def image_loader(image_name):              # 图像加载器
    image = Image.open(image_name)         # from PIL import Image，Image.open是按照RGB的顺序读入图像，cv2.imread是按照BGR
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)     # unsqueeze(0)  是用于升维的函数，将原来的图像升维为4维，以便于与minibatch相匹配，参数0表示在image的0维增加一个维度
    return image.to(device, torch.float)  # 将image的tensor拷贝一份到device所指向的GPU上，这样可以便于之后运算

basicpath = r'C:\Users\七翅兽\Desktop\【CSDN】Pytorch 图像风格迁移教程代码 逐句详解\images 4 neural-style'   # 这个地址换成您自己的文件地址
style_img = image_loader(basicpath + r'\picasso.jpg')      # 风格图
content_img = image_loader(basicpath + r'\dancing.jpg')    # 内容图

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"  # assert为用户自定义的错误检测语句，如果条件为真，不进行任何操作，如果条件为假，则输出报错语句

# 图像可视化 ###########################################

unloader = transforms.ToPILImage()  # 将tensor重新转换为PIL图像

plt.ion()          # 开启PLT绘图的交互模式，以便于输出多张图片

def imshow(tensor, title=None):  # 定义图像输出函数
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # 将图片降维，去掉之前添加的batchsize那一维
    image = unloader(image)       # 转换为PIL图像
    plt.imshow(image)             # 画图
    if title is not None:       # 如果输入title
        plt.title(title)
    plt.pause(0.001)              # pause a bit so that plots are updated


plt.figure()                                 # 新建figure
imshow(style_img, title='Style Image')       # 调用子函数，画图

plt.figure()
imshow(content_img, title='Content Image')

# 内容损失 ###########################################
class ContentLoss(nn.Module):             #  # import torch.nn as nn    表示Content是nn.Module的子类

    def __init__(self, target,):               # 定义构造方法
        super(ContentLoss, self).__init__()    # 此处self是Contentloss类，这句话将self转换为父类nn.Module类，然后调用父类的构造方法作为子类的构造方法
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()         # 将target从计算图中分离出来，使其不具备梯度

    def forward(self, input):                        # 前馈方法
        self.loss = F.mse_loss(input, self.target)   # 利用MSE计算输入图像与目标内容图像之间的损失
        return input

# gram矩阵 ###########################################
       # PS：假设输入图像经过卷积后，得到的feature map为[b, ch, h, w]。我们经过flatten和矩阵转置操作，
       # 可以变形为[b, ch, h*w]和[b, h*w, ch]的矩阵。再对1，2维作矩阵内积得到[b, ch, ch]大小的矩阵，这就是我们所说的Gram Matrices。
       # gram矩阵是计算每个通道I的feature map与每个通道j的feature map的内积。
       # gram matrix的每个值可以说是代表i通道的feature map与j通道的feature map的互相关程度。
       # 具体就是计算某一层，同个源图像得到所有特征图之间的关系（直接对应像素相乘求和），所以最后的形状应该为(b,ch,ch)
def gram_matrix(input):          # gram积用于保存图像的风格
    a, b, c, d = input.size()    # 读取input的size
                                 # a=batch size(=1)
                                 # b=number of feature maps
                                 # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # 计算gram 内积；   torch.mm是矩阵相乘，计算叉乘，torch.mul是计算点乘，compute the gram product

    # we 'normalize' the values of the gram matrix by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)     # 通过对gram积除以每一层的神经元数目，对其实现归一化。因为实际上我们更关注顶层在较大感知域带来的风格信息，归一化之后可以避免底层神经元较多，而放大底层风格对目标图像的影响




# 风格损失 ###########################################
class StyleLoss(nn.Module):     # import torch.nn as nn    表示Styleloss继承了nn.Module类

    def __init__(self, target_feature):                      # 子类定义了自己的构造函数
        super(StyleLoss, self).__init__()                    # 将styleloss类的对象self转换为父类nn.Modlue类，然后调用父类的构造函数，
                                                             # 目的是在子类的构造函数中调用父类的构造函数，并且在后面补充子类构造函数的特有成员
        self.target = gram_matrix(target_feature).detach()   # 计算target_feature的gram矩阵
    def forward(self, input):                      # Styleloss类中的forward方法，也就是风格损失的前馈方法
        G = gram_matrix(input)                     # 计算input的gram矩阵
        self.loss = F.mse_loss(G, self.target)     # import torch.nn.functional as F    使用mse度量目标风格的图片与输入图片之间的gram矩阵的mse损失
        return input

# 模型下载 ###########################################
cnn = models.vgg19(pretrained=True).features.to(device).eval()     # visual geometry group 19 ，载入VGG19的模型，大约500M

# 规范化模块 ###########################################

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)    # 样本均值
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)     # 样本标准差

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):                   # 类的继承
    def __init__(self, mean, std):
        super(Normalization, self).__init__()      # 继承父类的构造函数
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)    
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std     # 前馈过程中对样本进行归一化

# 将损失函数嵌入到模型中 ###########################################
# pytorch上下载的模型是将Conv2d, ReLU, MaxPool2d, Conv2d, ReLU…等多个子类序列化拼接在一起后的组成的。
# 因此，我们选定部分卷积层，将风格损失和内容损失的算子添加到选定的几个卷积层之后，
# 用户计算在输入图片与内容图片和风格图片造这些卷积层中所得到的feature map之间距离。

# desired depth layers to compute style/content losses :   选定一下几个卷积层进行计算
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)   # 深拷贝vgg19这个模型

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)   # 归一化模块

    # just in order to have an iterable access to or list of content/syle losses
    content_losses = []    # 内容损失
    style_losses = []      # 风格损失

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # nn.Sequential将会构造一个小型的序列模块，该模块可以进一步添加到已经构建好的网络
    model = nn.Sequential(normalization)     # 这句话构建里一个序列化模块，并且将normalization作为该模块的第一步

    i = 0                    # increment every time we see a conv，用于统计卷积层
    for layer in cnn.children():               # 依次遍历每个子层cnn.children()
        if isinstance(layer, nn.Conv2d):       # 判断当前layer是不是nn.Conv2d类
            i += 1                                  # 如果是，则i+1
            name = 'conv_{}'.format(i)              # 记录该层的名字
        elif isinstance(layer, nn.ReLU):       # 如果当前layer是nn.ReLU类
            name = 'relu_{}'.format(i)              # 记录该层的名字
            # The in-place version doesn't play very nicely with the ContentLoss and StyleLoss we insert below. So we replace with out-of-place ones here.
            layer = nn.ReLU(inplace=False) 
        elif isinstance(layer, nn.MaxPool2d):   # 如果当前layer是nn.MaxPool2d类 
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d): # 如果当前layer是nn.BatchNorm2d类 
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))   # RuntimeError调出计算机运行过程中的错误信息显示串口，进行报错

        model.add_module(name, layer)       # 将以上各层逐个添加到model这个模型中
                                            # model.add_module(name,layer)将一个模块加入到以名字name加入到model里

        if name in content_layers:        # 如果当前层属于内容层列表中
            # add content loss:
            target = model(content_img).detach()    # model(content_img)表示内容图片从model中前馈通过，.detach()表示将图片剥离出来，剔除其中的梯度信息
            content_loss = ContentLoss(target)      # 计算内容损失
            model.add_module("content_loss_{}".format(i), content_loss)    # 将内容损失添加到模型中
            content_losses.append(content_loss)     # 在内容损失列表中添加内容损失

        if name in style_layers:           # 如果是风格层
            # add style loss:
            target_feature = model(style_img).detach()   # 风格图片前馈
            style_loss = StyleLoss(target_feature)       # 风格损失
            model.add_module("style_loss_{}".format(i), style_loss)    # 将风格损失添加到该层中 
            style_losses.append(style_loss)              # 添加到风格损失列表中

    # now we trim off the layers after the last content and style losses
    # 将最后一个风格或者内容层之后的所有层都剪除
    for i in range(len(model) - 1, -1, -1):        # 从最后一层开始，反向遍历模型的每一层， range(start, stop[, step])
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):   # 当第一次遇到内容层或者损失层就break，
            break                                                                  # 也就是找到序列model中从底层到顶层的最后一个内容或者风格层
    model = model[:(i + 1)]     # 只截取model的0-i层，第i层就是最后一个内容或者风格层，相当于剪除了剩下的所有层

    return model, style_losses, content_losses   # 返回模型，风格损失，内容损失

# 输入样本 ###########################################
# 输入样本可以是白噪声图片，也可以是内容图片，一般为了减少计算负担，会选用内容图片

input_img = content_img.clone()       # 克隆一张内容图用作输入图
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
plt.figure()
imshow(input_img, title='Input Image')

# 优化器 ###########################################
def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])     # input_img.requires_grad_()表明，训练的时候向input_img施加梯度，对图片像素点进行调整
    return optimizer
# 训练函数 ###########################################
def run_style_transfer(cnn, normalization_mean, normalization_std,          # 定义训练函数
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)       # 调用子函数，构建模型
    optimizer = get_input_optimizer(input_img)                               # 优化器

    print('Optimizing..')
    run = [0]         # 迭代次数的计步器
    while run[0] <= num_steps:     # 迭代次数

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)   # 每次对输入图片进行训练调整后，图片中部分像素点会超出0-1的范围，因此要对其进行剪切

            optimizer.zero_grad()    # 每次epoch的时候将梯度置为0，原因是pytorch的梯度是逐次累加的，因此每次调用的时候就需要先置为0
            model(input_img)         # 前馈，将input_image输入模型
            style_score = 0          # 本次epoch的风格损失
            content_score = 0

            for sl in style_losses:           # 遍历所有的风格损失。 style_losses.append(style_loss)  将所有计算风格损失的算子的结果累加起来
                style_score += sl.loss        # 将所有层的风格损失相加
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight       # 风格损失乘以其权重
            content_score *= content_weight

            loss = style_score + content_score  # 最终损失函数是两者相加
            loss.backward()                     # 反馈

            run[0] += 1                         # 计步器+1
            if run[0] % 50 == 0:                # 每训练50次就在品目上打印一次结果
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score      # 返回风格损失与内容损失的结果

        optimizer.step(closure)      # 优化器对风格损失与内容损失，然后进行优化

    # a last correction...
    input_img.data.clamp_(0, 1)     # 将数据值压缩到0-1之间

    return input_img         # 最终输出输入图，该图就是风格迁移后的图

# 开启训练 ###########################################
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)                # 调用子函数进行训练

plt.figure()
imshow(output, title='Output Image')      # 画出最终风格迁移后的图

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()   # 图片输出

