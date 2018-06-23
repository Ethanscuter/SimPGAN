import os
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
# from baseline.train import softmax_pretrain_on_datasets
import numpy as np
from skimage import io
from scipy import misc
from numpy.random import randint, shuffle, choice
from PIL import Image
import torch.backends.cudnn as cudnn
from pretrain import Identify_net

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Hyper Parameters
input_size = 224
num_classes = 751
num_epochs = 300  # 30
steps_per_epoch = 33000  # 100
batch_size = 48
learning_rate = 0.001


class ReidDataPrepare():
    def __init__(self, data_list_path, train_dir_path):
        self.list = data_list_path
        self.dir = train_dir_path

    def return_class_img_labels(self):
        class_img_labels = dict()
        class_cnt = -1
        last_label = -2
        with open(self.list, 'r') as f:
            for line in f:
                line = line.strip()
                img = line
                lbl = int(line.split('_')[0])
                if lbl != last_label:
                    class_cnt = class_cnt + 1
                    cur_list = list()
                    class_img_labels[str(class_cnt)] = cur_list
                last_label = lbl

                img_name = os.path.join(self.dir, img)
                image = io.imread(img_name)
                image = misc.imresize(image, size=(224, 224))
                image = np.expand_dims(image, axis=0)

                class_img_labels[str(class_cnt)].append(image[0])

        return class_img_labels


class SiameseNetworkDataset(data.Dataset):
    def __init__(self, train_list, train_dir, transforms=None):
        # Initialize file path or list of file names.
        self.list_path = train_list
        self.root_dir = train_dir
        self.transform = transforms
        self.class_img_labels = ReidDataPrepare(self.list_path, self.root_dir).return_class_img_labels()

        self.left_images = list()
        self.right_images = list()

        self.label1 = list()
        self.label2 = list()

        self.cur_epoch = 0
        self.pos_prop = 4

        for m in range(int(steps_per_epoch / batch_size)):
            self.left_label = randint(len(self.class_img_labels), size=batch_size)
            if self.cur_epoch % self.pos_prop == 0:
                self.right_label = self.left_label
            else:
                self.right_label = np.copy(self.left_label)
                shuffle(self.right_label)
            slice_start = 0
            for i in range(batch_size):
                len_left_label_i = len(self.class_img_labels[str(self.left_label[i])])
                self.left_images.append(self.class_img_labels[str(self.left_label[i])]
                                        [int(slice_start * len_left_label_i):]
                                        [choice(len_left_label_i - int(len_left_label_i * slice_start))])
                len_right_label_i = len(self.class_img_labels[str(self.right_label[i])])
                self.right_images.append(self.class_img_labels[str(self.right_label[i])]
                                        [int(slice_start * len_right_label_i):]
                                        [choice(len_right_label_i - int(len_right_label_i * slice_start))])

            self.label1 += self.left_label.tolist()
            self.label2 += self.right_label.tolist()

            self.cur_epoch += 1

        self.label1 = np.array(self.label1, dtype=int)
        self.label2 = np.array(self.label2, dtype=int)
        # transform before the the construct dataset
        self.left_images = np.array(self.left_images, dtype=float)
        self.right_images = np.array(self.right_images, dtype=float)
        self.binary_label = (self.label1 == self.label2).astype(int)

    def __getitem__(self, index):
        # Read one data from file
        # Preprocess the data (torchvision.Transform)
        # Return a data pair (image and label)
        """
        Args:
            param index(int): Index

            Returns:
                tuple: (image, target) where target is index of the target class.
        """

        img_left, img_right, target_left, target_right, target_binary = self.left_images[index], \
                                                                        self.right_images[index], \
                                                                        self.label1[index], \
                                                                        self.label2[index], \
                                                                        self.binary_label[index]

        img_left = Image.fromarray(img_left.astype('uint8'))
        img_right = Image.fromarray(img_right.astype('uint8'))

        if self.transform is not None:
            img_left = self.transform(img_left)
            img_right = self.transform(img_right)

        # Before make samples, image and labels should be numpy data structure
        return img_left, img_right, target_left, target_right, target_binary

    def __len__(self):
        return len(self.left_images)


class Siamese(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.fc = nn.Linear(2048, 2)

    def forward(self, input1, input2):
        feature1, out1 = self.base_model(input1)
        feature2, out2 = self.base_model(input2)
        out = (feature1 - feature2) ** 2
        out = self.fc(out)

        return feature1, out1, out2, out


def pair_tune(source_model_path, tune_dataset, train_list, train_dir, num_classes):
    data_transform = transforms.Compose([
        transforms.Scale([286, 286], Image.BICUBIC),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # custom dataset
    pair_dataset = SiameseNetworkDataset(train_list=train_list,
                                         train_dir=train_dir,
                                         transforms=data_transform)

    # data loader
    train_loader = torch.utils.data.DataLoader(dataset=pair_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)

    # Model
    base_model = Identify_net(num_classes)
    base_model = torch.nn.DataParallel(base_model).cuda()
    cudnn.benchmark = True
    base_model.load_state_dict(torch.load(source_model_path))

    model = Siamese(base_model)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (image1, image2, label1, label2, label3) in enumerate(train_loader):
            # Convert torch tensor to Variable
            # labels = labels.cuda(async=True)
            label1 = label1.cuda(async=True)
            label2 = label2.cuda(async=True)
            label3 = label3.cuda(async=True)

            image1, image2 = Variable(image1), Variable(image2)
            label1, label2, label3 = Variable(label1), Variable(label2), Variable(label3)

            optimizer.zero_grad()
            feature, output1, output2, output3 = model(image1, image2)
            loss = criterion(output1, label1) + criterion(output2, label2) + criterion(output3, label3)
            loss.backward()
            optimizer.step()

            if (i+1) % 1 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch+1, num_epochs, i+1, len(pair_dataset)//batch_size, loss.data[0]))

    # Save model
    # torch.save(model, tune_dataset + '_pair_pretrain.pkl')
    torch.save(model.state_dict(), tune_dataset + '_pair_pretrain.pkl')


def pair_pretrain_on_dataset(source):
    # project_path = '/home/xintong/PRCGAN'  # Rename the project path
    project_path = '/home/wxt/PRCGAN'  # Rename the project path
    if source == 'market':
        train_list = project_path + '/market/market_train.list'  # Use identical list path
        train_dir = project_path + '/market/train'  # Use identical dir path
        class_count = 751
    # elif ...
    # class_img_labels = ReidDataPrepare(train_list, train_dir)

    pair_tune('./' + source + '_softmax_pretrain.pkl', source, train_list, train_dir, num_classes=class_count)


if __name__ == '__main__':
    sources = ['market']
    for source in sources:
        # softmax_pretrain_on_datasets(source)
        pair_pretrain_on_dataset(source)
