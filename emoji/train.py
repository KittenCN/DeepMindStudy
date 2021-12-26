from torch.utils.data import Dataset, DataLoader
import time
from torch.optim import AdamW, RMSprop, SGD, Adam
from model import *
from torchvision.utils import save_image
import random
from torch.autograd import Variable
import os
import cv2
from albumentations import Normalize, Compose, Resize, IAAAdditiveGaussianNoise, GaussNoise, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2
from apex import amp
import pickle

# ------------------------------------config------------------------------------
class config:
    # 设置种子数，配置是否要固定种子数
    seed = 26
    use_seed = False

    # 配置是否要从磁盘加载之前保存的模型参数继续训练
    from_old_model = False

    # 使用apex加速训练
    use_apex = True

    # 运行多少个epoch之后停止
    epochs = 20000
    # 配置batch size
    batchSize = 8

    # 每次保存模型时输出多少张样图
    save_img_size = 64

    # 训练图片输入分辨率，在训练前都预处理完成缩放
    img_size = 96

    # 配置喂入生成器的随机正态分布种子数有多少维（如果改动，需要在model中修改网络对应参数）
    img_seed_dim = 128

    # 有多大概率在训练判别器D时交换正确图片的标签和伪造图片的标签
    D_train_label_exchange = 0.1

    # 将数据集保存在内存中还是磁盘中
    # 小型数据集可以整个载入内存加快速度
    read_from = "Memory"
    # read_from = "Disk"

    # 保存模型参数文件的路径
    G_model_path = "emoji/model/G_model.pth"
    D_model_global_path = "emoji/model/D_model_global.pth"
    D_model_patch_path = "emoji/model/D_model_patch.pth"

    # 保存优化器参数文件的路径
    G_optimizer_path = "emoji/model/G_optimizer.pth"
    D_optimizer_global_path = "emoji/model/D_optimizer_global.pth"
    D_optimizer_patch_path = "emoji/model/D_optimizer_patch.pth"

    # 保存当前保存模型的历史总计训练epoch数
    epoch_record_path = "emoji/model/epoch_count.pkl"

    # 当连接大容量移动硬盘时，对每个版本文件都进行单独备份，以方便回退历史版本
    # 如果这个路径不存在，则什么都不做
    # extra_backup_path = 'F:/version12'
    extra_backup_path = ""

    # 损失函数
    # # 使用均方差损失函数
    # criterion = nn.MSELoss()
    # 使用二分类交叉熵损失函数
    criterion = nn.BCEWithLogitsLoss()

    # 多少个epoch之后保存一次模型
    save_step = 10

    # ------------------------------------路径配置------------------------------------
    # 数据集来源
    img_path = "emoji/data/train/"
    # 输出图片的文件夹路径
    output_path = "emoji/model/output_images/"

    # 如果继续训练，则读取之前进行过多少次epoch的训练
    if from_old_model:
        with open(epoch_record_path, "rb") as file:
            last_epoch_number = pickle.load(file)
    else:
        last_epoch_number = 0


# 固定随机数种子
def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if config.use_seed:
    seed_all(seed=config.seed)


# -----------------------------------transforms------------------------------------
def get_transforms():
    # 缩放分辨率并转换到0-1之间
    return Compose(
         [ HorizontalFlip(p=0.5),
         VerticalFlip(p=0.5),
         Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, p=1.0),
         ToTensorV2(p=1.0)]
    )


# ------------------------------------dataset------------------------------------
# 从磁盘读取数据的dataset
if config.read_from == "Disk":
    class image_dataset(Dataset):
        def __init__(self, file_list, img_path, transform):
            # files list
            self.file_list = file_list
            self.img_path = img_path
            self.transform = transform

        def __getitem__(self, index):
            image_path = self.img_path + self.file_list[index]
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(image=img)['image']
            return img

        def __len__(self):
            return len(self.file_list)

# 从内存读取数据的dataset
elif config.read_from == "Memory":
    class image_dataset(Dataset):
        def __init__(self, file_list, img_path, transform):
            self.imgs = []
            for file in file_list:
                image_path = img_path + file
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.transform = transform
                self.imgs.append(img)

        def __getitem__(self, index):
            img = self.imgs[index]
            img = self.transform(image=img)['image']
            return img

        def __len__(self):
            return len(self.imgs)


# ------------------------------------main------------------------------------
def main():
    # 如果可以使用GPU运算，则使用GPU，否则使用CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Use " + str(device))

    # 创建输出文件夹
    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)

    # 如果检测到额外存储路径存在，发出通报
    if os.path.exists(config.extra_backup_path):
        print(f"Extra backup path [{config.extra_backup_path}] exists, extra backup version will be saved.")

    # 创建dataset
    # create dataset
    file_list = None
    for path, dirs, files in os.walk(config.img_path, topdown=False):
        file_list = list(files)

    train_dataset = image_dataset(file_list, config.img_path, transform=get_transforms())
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batchSize, shuffle=True)

    # 从model中获取判别器D和生成器G的网络模型
    # 判别器分为global全局判别器与patch局部判别器
    G_model = get_G_model(config.from_old_model, device, config.G_model_path)
    D_model_global = get_D_model_global(config.from_old_model, device, config.D_model_global_path)
    D_model_patch = get_D_model_patch(config.from_old_model, device, config.D_model_patch_path)

    G_model.train()
    D_model_global.train()
    D_model_patch.train()

    # 定义G和D的优化器，此处使用AdamW优化器
    G_optimizer = Adam(G_model.parameters(), lr=1e-4)
    D_optimizer_global = Adam(D_model_global.parameters(), lr=3e-4)
    D_optimizer_patch = Adam(D_model_patch.parameters(), lr=3e-4)
    # D_optimizer_global = AdamW(D_model_global.parameters(), lr=3e-4, weight_decay=1e-6)
    # D_optimizer_global = RMSprop(D_model_global.parameters(), lr=3e-4, alpha=0.9)
    # D_optimizer_global = SGD(D_model_global.parameters(), lr=3e-4)
    # D_optimizer_patch = AdamW(D_model_patch.parameters(), lr=3e-4, weight_decay=1e-6)
    # D_optimizer_patch = RMSprop(D_model_patch.parameters(), lr=3e-4, alpha=0.9)
    # D_optimizer_patch = SGD(D_model_patch.parameters(), lr=3e-4)

    # 如果是读取之前训练的数据，则加载保存的优化器参数
    if config.from_old_model:
        G_optimizer.load_state_dict(torch.load(config.G_optimizer_path))
        D_optimizer_global.load_state_dict(torch.load(config.D_optimizer_global_path))
        D_optimizer_patch.load_state_dict(torch.load(config.D_optimizer_patch_path))

    # 损失函数
    criterion = config.criterion

    # 混合精度加速
    if config.use_apex:
        G_model, G_optimizer = amp.initialize(G_model, G_optimizer, opt_level="O1")
        D_model_global, D_optimizer_global = amp.initialize(D_model_global, D_optimizer_global, opt_level="O1")
        D_model_patch, D_optimizer_patch = amp.initialize(D_model_patch, D_optimizer_patch, opt_level="O1")

    # 记录训练时间
    train_start = time.time()

    # 定义标签，单值标签用于传统判别器，多值标签用于patch判别器
    # 定义真标签，使用标签平滑的策略，全0.9
    real_labels = Variable(torch.ones(config.batchSize, 1)-0.1).to(device)

    # 定义假标签，单向平滑，因此不对生成器标签进行平滑处理，全0
    fake_labels = Variable(torch.zeros(config.batchSize, 1)).to(device)

    # 开始训练的每一个epoch
    for epoch in range(config.epochs):
        print("start epoch "+str(epoch+1)+":")
        # 定义一些变量用于记录进度和损失
        batch_num = len(train_loader)
        D_loss_sum_global = 0
        D_loss_sum_patch = 0
        G_loss_sum = 0
        count = 0

        # 从dataloader中提取数据
        for index, images in enumerate(train_loader):
            count += 1
            # 将图片放入运算设备的内存
            images = images.to(device)

            # 记录真假标签是否被交换过
            exchange_labels = False

            # 有一定概率在训练判别器时交换label
            if random.uniform(0, 1) < config.D_train_label_exchange:
                real_labels, fake_labels = fake_labels, real_labels
                exchange_labels = True

            # 训练判断器D_global
            D_optimizer_global.zero_grad()
            # 将随机的初始数据喂入生成器生成假图像
            img_seeds = torch.randn(config.batchSize, config.img_seed_dim).to(device)
            fake_images = G_model(img_seeds)
            # 用真样本输入判别器
            real_output = D_model_global(images)
            # 对于数据集末尾的数据，长度不够一个batch size时需要去除过长的真实标签
            if len(real_labels) > len(real_output):
                D_loss_real = criterion(real_output, real_labels[:len(real_output)])
            else:
                D_loss_real = criterion(real_output, real_labels)
            # 用假样本输入判别器
            fake_output = D_model_global(fake_images)
            D_loss_fake = criterion(fake_output, fake_labels)
            # 将真样本与假样本损失相加，得到判别器的损失
            D_loss_global = D_loss_real + D_loss_fake
            D_loss_sum_global += D_loss_global.item()
            # 重置优化器
            D_optimizer_global.zero_grad()
            # 用损失更新判别器
            if config.use_apex:
                with amp.scale_loss(D_loss_global, D_optimizer_global) as scaled_loss:
                    scaled_loss.backward()
            else:
                D_loss_global.backward()
            D_optimizer_global.step()

            # 训练判断器D_patch
            D_optimizer_patch.zero_grad()
            # 将随机的初始数据喂入生成器生成假图像
            img_seeds = torch.randn(config.batchSize, config.img_seed_dim).to(device)
            fake_images = G_model(img_seeds)
            # 用真样本输入判别器
            real_output = D_model_patch(images)
            # # 对于数据集末尾的数据，长度不够一个batch size时需要去除过长的真实标签
            if len(real_labels) > len(real_output):
                D_loss_real = criterion(real_output, real_labels[:len(real_output)])
            else:
                D_loss_real = criterion(real_output, real_labels)
            # 用假样本输入判别器
            fake_output = D_model_patch(fake_images)
            D_loss_fake = criterion(fake_output, fake_labels)
            # 将真样本与假样本损失相加，得到判别器的损失
            D_loss_patch = D_loss_real + D_loss_fake
            D_loss_sum_patch += D_loss_patch.item()
            # 重置优化器
            D_optimizer_patch.zero_grad()
            # 用损失更新判别器
            if config.use_apex:
                with amp.scale_loss(D_loss_patch, D_optimizer_patch) as scaled_loss:
                    scaled_loss.backward()
            else:
                D_loss_patch.backward()
            D_optimizer_patch.step()

            # 如果之前交换过真假标签，此时再换回来
            if exchange_labels:
                real_labels, fake_labels = fake_labels, real_labels

            # 将随机种子数喂入生成器G生成假数据
            img_seeds = torch.randn(config.batchSize, config.img_seed_dim).to(device)
            fake_images = G_model(img_seeds)
            # 将假数据输入判别器
            fake_output_global = D_model_global(fake_images)
            fake_output_patch = D_model_patch(fake_images)
            # 将假数据的判别结果与真实标签对比得到损失
            G_loss_global = criterion(fake_output_global, real_labels)
            G_loss_patch = criterion(fake_output_patch, real_labels)
            G_loss = G_loss_global + G_loss_patch
            G_loss_sum += G_loss.item()
            # 重置优化器
            G_optimizer.zero_grad()
            # 利用损失更新生成器G
            if config.use_apex:
                with amp.scale_loss(G_loss, G_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                G_loss.backward()
            G_optimizer.step()

            # 打印程序工作进度
            print("\rEpoch: %2d, Batch: %4d / %4d" % (epoch + 1, index + 1, batch_num), end="")

        print()
        if (config.last_epoch_number+epoch+1) % config.save_step == 0:
            print("Start saving model files to current path...", end='')
            # 在每N个epoch结束时保存模型参数到磁盘文件
            torch.save(G_model.state_dict(), config.G_model_path)
            torch.save(D_model_global.state_dict(), config.D_model_global_path)
            torch.save(D_model_patch.state_dict(), config.D_model_patch_path)
            # 在每N个epoch结束时保存优化器参数到磁盘文件
            torch.save(G_optimizer.state_dict(), config.G_optimizer_path)
            torch.save(D_optimizer_global.state_dict(), config.D_optimizer_global_path)
            torch.save(D_optimizer_patch.state_dict(), config.D_optimizer_patch_path)
            # 保存历史训练总数
            with open(config.epoch_record_path, "wb") as file:
                pickle.dump(config.last_epoch_number + epoch + 1, file, 1)
            # 在每N个epoch结束时输出一组生成器产生的图片到输出文件夹，拼接出一张含有config.save_img_size张图的大图
            save_imgs = []
            with torch.no_grad():
                for _ in range(config.save_img_size // config.batchSize):
                    img_seeds = torch.randn(config.batchSize, config.img_seed_dim).to(device)
                    fake_images = G_model(img_seeds).cuda().data
                    # 将假图像缩放到[0,1]的区间
                    fake_images = 0.5 * (fake_images + 1)
                    fake_images = fake_images.clamp(0, 1)
                    # 连接所有生成的图片然后用自带的save_image()函数输出到磁盘文件
                    fake_images = fake_images.view(-1, 3, config.img_size, config.img_size)
                    save_imgs.append(fake_images)
            save_imgs = torch.cat(save_imgs, 0)
            save_image(save_imgs, config.output_path+str(config.last_epoch_number + epoch + 1)+'.png')
            print("Success.")

            # 当连接大容量移动硬盘时，对每个版本文件都进行单独备份，以方便取出历史版本
            if os.path.exists(config.extra_backup_path):
                print("Start saving model files to extra backup path...", end='')
                extra_backup_path = f'{config.extra_backup_path}/{config.last_epoch_number + epoch + 1}/'
                # 创建该版本的历史总epoch存放目录
                if not os.path.exists(extra_backup_path):
                    os.mkdir(extra_backup_path)
                # 保存模型参数到磁盘文件
                torch.save(G_model.state_dict(), extra_backup_path + config.G_model_path)
                torch.save(D_model_global.state_dict(), extra_backup_path + config.D_model_global_path)
                torch.save(D_model_patch.state_dict(), extra_backup_path + config.D_model_patch_path)
                # 保存优化器参数到磁盘文件
                torch.save(G_optimizer.state_dict(), extra_backup_path + config.G_optimizer_path)
                torch.save(D_optimizer_global.state_dict(), extra_backup_path + config.D_optimizer_global_path)
                torch.save(D_optimizer_patch.state_dict(), extra_backup_path + config.D_optimizer_patch_path)
                # 保存历史训练总数
                with open(extra_backup_path + config.epoch_record_path, "wb") as file:
                    pickle.dump(config.last_epoch_number + epoch + 1, file, 1)
                # 保存对应版本的预览图片
                save_image(save_imgs, extra_backup_path + str(config.last_epoch_number + epoch + 1) + '.png')
                print("Success.")

        # 打印该epoch的损失，时间等数据用于参考
        print("D_loss_global:", round(D_loss_sum_global / count, 3))
        print("D_loss_patch:", round(D_loss_sum_patch / count, 3))
        print("G_loss:", round(G_loss_sum / count, 3))
        current_time = time.time()
        pass_time = int(current_time - train_start)
        time_string = str(pass_time // 3600) + " hours, " + str((pass_time % 3600) // 60) + " minutes, " + str(
            pass_time % 60) + " seconds."
        print("Time pass:", time_string)
        print()

    # 运行结束
    print("Done.")


if __name__ == '__main__':
    main()

