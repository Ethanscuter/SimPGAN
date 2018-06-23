import os
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from skimage import io
from PIL import Image
import torch.backends.cudnn as cudnn
from scipy import misc

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Hyper Parameters
input_size = 224
num_classes = 732
num_epochs = 90
# batch_size = 32
batch_size = 16
learning_rate = 0.001


class PersonReidDataset(data.Dataset):
    """Person Reid dataset

    Args:
        train_list(string): Directory of dataset label .list file
        train_dir(string): Directory of dataset where consists of person images
        transform(callable, optional): A function/transform that takes in an PIL image and returns a transform version.
        target_transform(callable,optional): A function/transform that takes in the target and transform it.

    """

    def __init__(self, train_list, train_dir, transform=None):
        # Initialize file path or list of file name

        self.list_path = train_list
        self.root_dir = train_dir
        self.transform = transform

        # now load the picked numpy arrarys
        self.images = []
        self.labels = []
        with open(self.list_path, 'r') as f:
            last_label = -1
            label_cnt = -1
            for line in f:
                line = line.strip()
                img = line
                lbl = line.split('_')[0]
                if last_label != lbl:
                    label_cnt += 1
                last_label = lbl
                img_name = os.path.join(self.root_dir, img)
                image = io.imread(img_name)
                image = misc.imresize(image, size=(224, 224))
                image = np.expand_dims(image, axis=0)

                self.images.append(image[0])
                self.labels.append(label_cnt)

        self.images = np.array(self.images, dtype=float)
        self.labels = np.array(self.labels, dtype=int)

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
        img, target = self.images[index], self.labels[index]

        img = Image.fromarray(img.astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        # Before make samples, image and labels should be numpy data structure
        return img, target

    def __len__(self):
        return len(self.images)


class Identify_net(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = True
        self.feature = nn.Sequential(*list(resnet.children())[:-1])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, class_count)

    def forward(self, x):
        x = self.feature(x)
        x_tmp = x.view(x.size(0), -1)
        x = self.dropout(x_tmp)
        out = self.fc(x)
        return x_tmp, out


def softmax_model_pretrain(train_list, train_dir, class_count, target_model_path):
    data_transform = transforms.Compose([
        transforms.Scale([286, 286], Image.BICUBIC),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    market_dataset = PersonReidDataset(train_list=train_list,
                                       train_dir=train_dir,
                                       transform=data_transform)

    train_loader = torch.utils.data.DataLoader(dataset=market_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    model = Identify_net(class_count)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            labels = labels.cuda(async=True)

            images = Variable(images)
            labels = Variable(labels)

            optimizer.zero_grad()
            out_nouse, outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch+1, num_epochs, i+1, len(market_dataset)//batch_size, loss.data[0]))

    correct = 0
    total = 0
    for images, labels in train_loader:
        images = Variable(images)
        out_nouse, outputs = model(images)
        outputs = outputs.cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

    # Save model
    # torch.save(model, target_model_path)

    torch.save(model.state_dict(), target_model_path)


def softmax_pretrain_on_datasets(source):
    # project_path = '/home/xintong/PRCGAN'  # Rename the project path
    project_path = '/home/wxt/PRCGAN'
    if source == 'market':
        train_list = project_path + '/market/market_train.list'  # Use identical list path
        train_dir = project_path + '/market/train'  # Use identical dir path
        class_count = 751
    # elif ...

    softmax_model_pretrain(train_list, train_dir, class_count, './' + source + '_softmax_pretrain.pkl')


if __name__ == '__main__':
    sources = ['market']
    for source in sources:
        softmax_pretrain_on_datasets(source)



