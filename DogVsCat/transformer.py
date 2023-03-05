import torch
import os
import glob
import platform
import zipfile
import numpy as np
import sys
import random

from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 100
batch_size = 16
base_dir = sys.path[0]
train_dir = r'data\DogVsCat\data\cats_and_dogs_small\train'
test_dir = r'data\DogVsCat\data\cats_and_dogs_small\test'
animal_list = ['cats', 'dogs']

train_transform = transforms.Compose(
    [
        transforms.Resize((244, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        if platform.system() == "Windows":
            splitsig = "\\"
        else:
            splitsig = "/"
        label = img_path.split(splitsig)[-1].split(".")[0]
        if label == "cat":
            label = 0
        else:
            label = 1
        return img_transformed, label

def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def getDataLoader(mode='train'):
    if mode == 'train':
        train_list, valid_list = PreProcess(mode)
        train_dataset = CatsDogsDataset(train_list, transform=train_transform)
        valid_dataset = CatsDogsDataset(valid_list, transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader
    else:
        test_list = PreProcess(mode)
        test_dataset = CatsDogsDataset(test_list, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader

def ExtractData():
    print("Extracting data...")
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    with zipfile.ZipFile(os.path.join(base_dir, "data", "train.zip")) as train_zip:
        train_zip.extractall(os.path.join(base_dir, "data"))
    with zipfile.ZipFile(os.path.join(base_dir, "data", "test.zip")) as test_zip:
        test_zip.extractall(os.path.join(base_dir, "data"))
    print("Extracting data done.")

def PreProcess(mode, RandomPlots=False):
    print("Loading data...")
    if mode == "train":
        train_list = []
        for item in animal_list:
            train_list += (glob.glob(os.path.join(train_dir + '\\' + item, "*.jpg")))
        print("Loading train data done.")
        if platform.system() == "Windows":
            splitsig = "\\"
        else:
            splitsig = "/"
        labels = [path.split(splitsig)[-1].split(".")[0] for path in train_list]

        if RandomPlots:
            random_idx = np.random.randint(1, len(train_list), 9)
            fig, axes = plt.subplots(3, 3, figsize=(16, 12))
            plt.subplots_adjust(hspace=0.5)
            for idx, ax in enumerate(axes.ravel()):
                img = Image.open(train_list[random_idx[idx]])
                ax.imshow(img)
                ax.set_title(str(random_idx[idx]) + " " + labels[random_idx[idx]])
            plt.show()
        
        train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels, random_state=42)
        return train_list, valid_list
    else:
        test_list = []
        for item in animal_list:
            test_list += (glob.glob(os.path.join(test_dir + '\\' + item, "*.jpg")))
        print("Loading test data done.")
        if RandomPlots:
            random_idx = np.random.randint(1, len(test_list), 9)
            fig, axes = plt.subplots(3, 3, figsize=(16, 12))
            plt.subplots_adjust(hspace=0.5)
            for idx, ax in enumerate(axes.ravel()):
                img = Image.open(test_list[random_idx[idx]])
                ax.imshow(img)
                ax.set_title(str(random_idx[idx]))
            plt.show()
        return test_list
        

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  # 进入前LN正则化,接着attention层
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))                        # 进入前LN正则化,接着FFN层
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x     # 残差连接Add
            x = ff(x) + x       # 残差连接Add
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width   # token纬度,论文page3
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(    # 论文中page3蓝色字: X*E=[196x768]*[768x768]=[196x768]
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),              # 架构图中 Linear Projection of Flattened Patches
        )
        # 与transformer论文不同的点,可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))     # 论文中[1 x (196+1) x 768]
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))                       # 论文中[1x1x768]
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)   # transformer论文的编码器

        self.pool = pool                    # cls汇聚方式:             数据使用cls[batch x数据第0号token]
        self.to_latent = nn.Identity()      # 不区分参数的占位符标识运算符,可以放到残差网络里就是在跳过连接的地方用这个层,此处仅占位

        self.mlp_head = nn.Sequential(      # 解码器,由于任务简单,使用mlp,数据使用cls[batch x数据第0号token]
            nn.LayerNorm(dim),              # 论文架构图 橙色MLP Head
            nn.Linear(dim, num_classes)     # 论文架构图 橙色Class
        )

    def forward(self, img):
        """

        :param img: [B C H W]
        :return:
        """
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape   # 论文中x=[Bx196x768]

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)  # 论文中cls=[Bx1x768]
        x = torch.cat((cls_tokens, x), dim=1)                       # 论文中XE=[(196+1)x768],x=[Bx(196+1)x768]
        x += self.pos_embedding[:, :(n + 1)]                        # 直接加
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
if __name__ == '__main__':
    seed_everything(42)

    model = ViT(image_size=224, patch_size=batch_size, num_classes=2, dim=128, depth=12, heads=16, mlp_dim=2048).to(device)
    print(model)
    criterionFun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    schedulerFun = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    print("train start")
    train_loader, val_loader = getDataLoader('train')
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        with tqdm(total=len(train_loader), ncols=100) as t:
            t.set_description(f'Epoch {epoch + 1}/{epochs}')
            for data, label in train_loader:
                t.update()
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterionFun(output, label)
                loss.backward()
                optimizer.step()
                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_loss += loss / len(train_loader)
                epoch_accuracy += acc / len(train_loader)
            with torch.no_grad():
                val_loss = 0
                val_acc = 0
                for data, label in val_loader:
                    data, label = data.to(device), label.to(device)
                    output = model(data)
                    loss = criterionFun(output, label)
                    acc = (output.argmax(dim=1) == label).float().mean()
                    val_loss += loss / len(val_loader)
                    val_acc += acc / len(val_loader)
            print(f'Epoch {epoch + 1}/{epochs} train loss: {epoch_loss:.4f} train acc: {epoch_accuracy:.4f} val loss: {val_loss:.4f} val acc: {val_acc:.4f}')
            schedulerFun.step()
        torch.save(model.state_dict(), f'./model/vit_{batch_size}.pth')
        print('model saved')

    print("test start")
    model = ViT(image_size=224, patch_size=batch_size, num_classes=2, dim=128, depth=12, heads=12, mlp_dim=2048).to(device)
    model.load_state_dict(torch.load('./model/vit_{batch_size}.pth', map_location=device))
    test_loader = getDataLoader('test')
    result = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            result.append(output.argmax(dim=1).detech().numpy())
    result = np.concatenate(result)
    print(result)
    print("test end")
    