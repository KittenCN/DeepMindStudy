import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL.Image as pil_image
import matplotlib.pyplot as plt

classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']

colormap = [[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],
            [192,0,0],[64,128,0],[192,128,0],[64,0,128],
            [192,0,128],[64,128,128],[192,128,128],[0,64,0],
            [128,64,0],[0,192,0],[128,192,0],[0,64,128]]

cm2lbl = np.zeros(256**3)
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i

def image2label(image):
    data = np.array(image, dtype='int32')
    idx = (data[:,:,0]*256+data[:,:,1])*256+data[:,:,2]
    return np.array(cm2lbl[idx], dtype='int64')

# image = pil_image.open(r'data\\VOC2012\\JPEGImages\\2007_000032.jpg').convert('RGB')
# image = transforms.RandomCrop((224,224))(image)
# print(image)
# plt.imshow(image)
# label = transforms.FixCrop((224,224))(pil_image.open(r'data\\VOC2012\\SegmentationClass\\2007_000032.png'))

class VOCDataset(Dataset):
    def __init__(self, file_path=None, transform=None):
        images_labels = []
        file = open(file_path, 'r')
        for name in file.readlines():
            name = name.strip()
            image_path = r'data\\VOC2012\\JPEGImages\\' + name + '.jpg'
            label_path = r'data\\VOC2012\\SegmentationClass\\' + name + '.png'
            images_labels.append((image_path, label_path))
        self.images_labels = images_labels
        self.transform = transform
    
    def __getitem__(self, index):
        image_path, label_path = self.images_labels[index]
        image = pil_image.open(image_path).convert('RGB')
        label = pil_image.open(label_path).convert('RGB')
        image = transforms.Resize((512, 512))(image)
        label = transforms.Resize((512, 512))(label)
        if self.transform is not None:
            image = self.transform(image)
        label = image2label(label)
        label = torch.from_numpy(label)
        return image, label

    def __len__(self):
        return len(self.images_labels)

transforms_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
])

train_dataset = VOCDataset(file_path=r'data\\VOC2012\\ImageSets\\Segmentation\\train.txt', transform=transforms_train)
test_dataset = VOCDataset(file_path=r'data\\VOC2012\\ImageSets\\Segmentation\\val.txt', transform=transforms_test)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, sampler=None)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, sampler=None)

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        n_class = 21
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1, stride=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)

        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1, stride=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)

        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1, stride=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1, stride=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1, stride=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)

        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, stride=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)

        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1, stride=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)

        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1, stride=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)

        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1, stride=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)

        self.maxpool4 = nn.MaxPool2d(2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1, stride=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)

        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1, stride=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)

        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1, stride=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)

        self.maxpool5 = nn.MaxPool2d(2, stride=2)

        self.conv6 = nn.Conv2d(512, 4096, 7, padding=1, stride=1)
        self.bn6 = nn.BatchNorm2d(4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d(p=0.5)

        self.conv7 = nn.Conv2d(4096, 4096, 1, padding=1, stride=1)
        self.bn7 = nn.BatchNorm2d(4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(p=0.5)

        self.conv8 = nn.Conv2d(4096, n_class, 1, padding=1, stride=1)

        # output = ((input -1) * stride) - 2 * padding + kernel_size + output_padding
        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, 2, stride=2, bias=False)
        self.pool4_conv = nn.Conv2d(512, n_class, 1, stride=1)
        self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, 2, stride=2, bias=False)
        self.pool3_conv = nn.Conv2d(256, n_class, 1, stride=1)
        self.upscore8 = nn.ConvTranspose2d(n_class, n_class, 8, stride=8, bias=False)

    def forward(self, x):
        h = x

        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h = self.pool1(h)

        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        h = self.pool2(h)

        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        h = self.maxpool3(h)
        pool3 = h

        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        h = self.maxpool4(h)
        pool4 = h

        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        h = self.maxpool5(h)

        h = self.relu6(self.bn6(self.conv6(h)))
        h = self.drop6(h)

        h = self.relu7(self.bn7(self.conv7(h)))
        h = self.drop7(h)

        h = self.conv8(h)
        h = self.upscore2(h)
        up_conv8 = h

        h2 = self.pool4_conv(pool4)
        h2 = up_conv8 + h2
        h2 = self.upscore_pool4(h2)
        up_pool4 = h2

        h3 = self.pool3_conv(pool3)
        h3 = up_pool4 + h3

        h3 = self.upscore8(h3)
        return h3

class SegementationMetric():
    def __init__(self, n_class):
        self.n_class = n_class
        self.confusion_matrix = np.zeros((n_class, n_class))
    
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusion_matrix += self.getCofusionMatrix(imgPredict, imgLabel)
        return self.confusion_matrix
    
    def getCofusionMatrix(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        mask = (imgLabel >= 0) & (imgLabel < self.n_class)
        label = self.n_class * imgLabel[mask].astype('int') + imgPredict[mask]
        count = np.bincount(label, minlength=self.n_class**2)
        confusion_matrix = count.reshape(self.n_class, self.n_class)
        return confusion_matrix

    def pixelAccuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def classPixelAccuracy(self):
        cpa = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return cpa
    
    def meanPixelAccuracy(self):
        mpa = np.nanmean(self.classPixelAccuracy())
        return mpa
    
    def intersectionOverUnion(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusionMaterix, axis=1) + np.sum(self.confusionMatrix, axis=0) - intersection
        iou = intersection / union
        return iou

    def meanIntersectionOverUnion(self):
        miou = np.nanmean(self.intersectionOverUnion())
        return miou

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_class, self.n_class))

model = FCN()
print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 100

lossfunction = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.to(device)
for epoch in range(epochs):
    loss_add = 0
    pa_add = 0
    mpa_add = 0
    miou_add = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = lossfunction(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_add += loss.data.item()
        pa_add += SegementationMetric.pixelAccuracy()
        mpa_add += SegementationMetric.meanPixelAccuracy()
        miou_add += SegementationMetric.meanIntersectionOverUnion()
    epoch_loss = loss_add / len(train_loader)
    epoch_pa = pa_add / len(train_loader)
    epoch_mpa = mpa_add / len(train_loader)
    epoch_miou = miou_add / len(train_loader)
    print('Epoch: {}/{}...'.format(epoch+1, epochs),
            'Loss: {:.4f}...'.format(epoch_loss),
            'Pixel Accuracy: {:.4f}...'.format(epoch_pa),
            'Mean Pixel Accuracy: {:.4f}...'.format(epoch_mpa),
            'Mean Intersection Over Union: {:.4f}...'.format(epoch_miou))
