import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import platform
from torchvision import transforms
from PIL import Image
from skimage import img_as_float
class GetDataSet():
    def __init__(self, dataSetName):
        self.dataSetName = dataSetName

        self.train_data = None
        self.train_label = None
        self.train_datasize = None

        self.test_data = None
        self.test_label = None
        self.test_datasize = None

        if self.dataSetName == 'MNIST' or self.dataSetName == 'mnist':
            self.mnistDataDistribution()
            print("mnist!!")
        elif self.dataSetName == 'EMNIST' or self.dataSetName == 'emnist':
            self.emnistDataDistribution()
            print("Emnist!!")
        elif self.dataSetName == 'CIFAR10' or self.dataSetName == 'cifar10':
            self.cifar10DataDistribution()
            print("cifar10!!")
        elif self.dataSetName == 'FASHIONMNIST' or self.dataSetName == 'fashionmnist':
            self.fashionmnistDataDistribution()
            print("fashion!!")
        elif self.dataSetName == 'imagenette2' or self.dataSetName == 'IMAGENETTE2':
            self.imagenette2DataDistribution()
    
    def imagenette2DataDistribution(self):
        data_dir = 'data/imagenette2'
        train_images_path = os.path.join(data_dir, 'train')
        test_images_path = os.path.join(data_dir, 'val')
        print('Preparing dataset ...')
        print('获取训练数据 ...')
        train_img = []
        train_label= []
        labels = os.listdir(train_images_path)
        for i, label in enumerate(labels): 
            img_files = os.listdir(train_images_path + '/' + label)
            train_img += img_files
            train_label += [i] * len(img_files)
        print('训练集:共 %d 张' % len(train_img))
        train_images, train_labels = self.get_train_images(train_img, train_label, train_images_path)
        self.train_data = train_images.astype(np.float32)
        self.train_label= torch.tensor(train_labels).to(torch.int64)
        self.train_datasize = len(self.train_data)
        print('训练数据准备完毕')

        print('获取测试数据 ...')
        test_img = []
        test_label = []
        labels = os.listdir(test_images_path)
        for i, label in enumerate(labels): 
            img_files = os.listdir(test_images_path + '/' + label + '/')
            test_img += img_files
            test_label += [i] * len(img_files)
        print('测试集:共 %d 张' % len(test_img))
        test_images, test_labels = self.get_test_images(test_img, test_label, test_images_path)
        # test_images = np.multiply(test_images, 1.0 / 255.0)
        self.test_data = test_images.astype(np.float32)
        self.test_label = torch.tensor(test_labels).to(torch.int64)
        self.test_datasize = len(self.test_data)
        print('测试数据准备完毕')
        print("self.train_label", self.train_label)
        pass 
    def get_train_images(self, train_img, train_label, train_images_path):
        labels = os.listdir(train_images_path)
        train_images = []
        for i, img_file in enumerate(train_img):
            img_dir = train_images_path + '/' + labels[train_label[i]] + '/'
            img_3_32_32 = self.extract(img_dir + img_file)
            train_images.append(img_3_32_32)
            if (i + 1) % 1000== 0:
                print('已完成%d张图片' % (i+1))
        return np.array(train_images), np.array(train_label)
   
    def get_test_images(self, test_img, test_label, test_images_path):
        labels = os.listdir(test_images_path)
        test_images = []
        for i, img_file in enumerate(test_img):
            img_dir = test_images_path + '/' + labels[test_label[i]] + '/'
            img_3_32_32 = self.extract(img_dir + img_file)
            test_images.append(img_3_32_32)
            if (i + 1) % 100== 0:
                print('已完成%d张图片' % (i+1))
        return np.array(test_images), np.array(test_label)
        
    def extract(self, image_file):
        img = Image.open(image_file)
        # print("img", img)

        img = img.resize((32, 32))
        # print("Image mode:", img.mode)
        if img.mode == 'L':
            img = img.convert('RGB')
        # img_3d = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = img_as_float(np.array(img))
        img = np.array(img)
        # print(img)
        img = np.transpose(img, (2, 0, 1))
        return img
    


    # def mnistDataDistribution(self, isIID):
    #
    #     trainingData = datasets.CIFAR10(
    #         root="data",
    #         train=True,
    #         download=True,
    #         transform=ToTensor(),
    #     )
    #     train_data = []
    #     train_label = []
    #     for X, y in trainingData:
    #         train_data.append(X.tolist())
    #         train_label.append(y)
    #     self.train_datasize = len(train_data)
    #     # ----------------------------------------------------------- #
    #     testingData = datasets.CIFAR10(
    #         root="data",
    #         train=False,
    #         download=True,
    #         transform=ToTensor(),
    #     )
    #     test_data = []
    #     test_label = []
    #     for X, y in testingData:
    #         test_data.append(X.tolist())
    #         test_label.append(y)
    #     self.test_datasize = len(test_data)
    #     self.test_data = torch.tensor(test_data)
    #     self.test_label = torch.tensor(test_label)
    #     # ----------------------------------------------------------- #
    #
    #     if isIID == True:
    #         self.train_data = torch.tensor(train_data)
    #         self.train_label = torch.tensor(train_label)
    #         print(1)
    #
    #     else:
    #         train_dataT = np.array(train_data, dtype='float32')
    #         train_labelT = np.array(train_label, dtype='int64')
    #         self.train_data = train_dataT
    #         self.train_label = train_labelT
    #     print(self.train_data.shape)

    def emnistDataDistribution(self, ):
        data_dir = r'./data/EMNIST'
        train_images_path = os.path.join(data_dir, 'emnist-balanced-train.csv')
        test_images_path = os.path.join(data_dir, 'emnist-balanced-test.csv.gz')
        import pandas as pd
        import numpy as np

        # 读取 CSV 文件
        train_images = pd.read_csv(train_images_path)
        test_images = pd.read_csv(train_images_path)
        # 提取标签
        train_labels = train_images.iloc[:, 0].values
        test_labels = test_images.iloc[:, 0].values
        # 提取图像数据并转换为 numpy 数组
        train_images = train_images.iloc[:, 1:].values
        train_images = train_images.astype(np.float32)  # 将图像数据转换为 float32 类型
        train_images = np.reshape(train_images, (-1, 1, 28, 28))  # 将图像数据重新整形为 28x28 的数组

        test_images = test_images.iloc[:, 1:].values
        test_images = test_images.astype(np.float32)  # 将图像数据转换为 float32 类型
        test_images = np.reshape(test_images , (-1,  1,  28, 28))  # 将图像数据重新整形为 28x28 的数组
        # 打印标签和图像的形状
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        self.train_data = train_images
        self.train_label = train_labels
        self.test_data = test_images
        self.test_label = test_labels
        print(self.train_data.shape)
        print(self.train_label.shape)

        balance_test_data = []
        balance_test_label = []
        class_index = [np.argwhere(self.test_label == y).flatten() for y in range(self.test_label.max() + 1)]
        min_number = min([len(class_) for class_ in class_index])
        for number in range(self.test_label.max() + 1):
            balance_test_data.append(self.test_data[class_index[number][:min_number]])
            balance_test_label += [number] * min_number
        print(min_number)
        self.test_data = np.concatenate(balance_test_data, axis=0)
        self.test_label = np.array(balance_test_label)
        self.test_label = torch.tensor(self.test_label).to(torch.int64)
    def mnistDataDistribution(self, ):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = 'data/MNIST/raw'
        data_dir = os.path.join(current_dir, data_dir)
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = self.extract_images(train_images_path)

        # print(train_images.shape) # 图片的形状 (60000, 28, 28, 1) 60000张 28 * 28 * 1  灰色一个通道
        # print('-' * 22 + "\n")
        train_labels = self.extract_labels(train_labels_path)
        # print("-" * 5 + "train_labels" + "-" * 5)
        # print(train_labels.shape)  # label shape (60000, 10)
        # print('-' * 22 + "\n")
        test_images = self.extract_images(test_images_path)
        test_labels = self.extract_labels(test_labels_path)


        # assert train_images.shape[0] == train_labels.shape[0]
        # assert test_images.shape[0] == test_labels.shape[0]
        #
        #
        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]
        #
        # assert train_images.shape[3] == 1
        # assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], 1, test_images.shape[1], test_images.shape[2])

        train_images = train_images.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        # print(train_images[0:10,5:10])
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        self.train_data = train_images
        self.train_label = np.argmax(train_labels == 1, axis = 1)
        self.test_data = test_images
        self.test_label = np.argmax(test_labels == 1, axis = 1)
        print(self.train_data.shape)
        balance_test_data = []
        balance_test_label = []
        class_index = [np.argwhere(self.test_label == y).flatten() for y in range(self.test_label.max() + 1)]
        min_number = min([len(class_) for class_ in class_index])
        for number in range(self.test_label.max() + 1):
            balance_test_data.append(self.test_data[class_index[number][:min_number]])
            balance_test_label += [number] * min_number

        self.test_data = np.concatenate(balance_test_data, axis=0)
        self.test_label = np.array(balance_test_label)
        self.test_label = torch.tensor(self.test_label).to(torch.int64)



    def fashionmnistDataDistribution(self, ):
        print("执行了吗？")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = 'data/FashionMNIST/raw'
        data_dir = os.path.join(current_dir, data_dir)
        #data_dir = r'./data/FashionMNIST/raw'
        
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = self.extract_images(train_images_path)

        # print(train_images.shape) # 图片的形状 (60000, 28, 28, 1) 60000张 28 * 28 * 1  灰色一个通道
        # print('-' * 22 + "\n")
        train_labels = self.extract_labels(train_labels_path)
        # print("-" * 5 + "train_labels" + "-" * 5)
        # print(train_labels.shape)  # label shape (60000, 10)
        # print('-' * 22 + "\n")
        test_images = self.extract_images(test_images_path)
        test_labels = self.extract_labels(test_labels_path)


        # assert train_images.shape[0] == train_labels.shape[0]
        # assert test_images.shape[0] == test_labels.shape[0]
        #
        #
        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]
        #
        # assert train_images.shape[3] == 1
        # assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], 1, test_images.shape[1], test_images.shape[2])

        train_images = train_images.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        # print(train_images[0:10,5:10])
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        print("train_labels", train_labels)
        self.train_data = train_images

        self.train_label = np.argmax(train_labels == 1, axis = 1)
        self.test_data = test_images
        self.test_label = np.argmax(test_labels == 1, axis = 1)
        print(self.train_data.shape)
        print(self.train_label.shape)


    def extract_images(self, filename):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:
                raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
            num_images = self._read32(bytestream)
            rows = self._read32(bytestream)
            cols = self._read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            return data

    def _read32(self, bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')

        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    def extract_labels(self, filename):
        """Extract the labels into a 1D uint8 numpy array [index]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return self.dense_to_one_hot(labels)

    def dense_to_one_hot(self, labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def cifar10DataDistribution(self):
        cifar10_dir = 'data/cifar-10-batches-py'
        print(self.train_label)
        self.train_data, self.train_label, self.test_data, self.test_label = self.load_CIFAR10(cifar10_dir)

    def load_CIFAR10(self, ROOT):
        """ load all of cifar """
        xs = []
        ys = []
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b,))
            X, Y = self.load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del xs, ys
        Xte, Yte = self.load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

        X_train = np.multiply(Xtr, 1.0 / 255.0)
        X_test = np.multiply(Xte, 1.0 / 255.0)
        # Resize images to 224x224

        # X_train = Xtr
        # X_test = Xte
        # X_train = torch.Tensor(Xtr).permute(0, 1, 2, 3) / 255.0
        # X_test = torch.Tensor(Xte).permute(0, 1, 2, 3) / 255.0
        return X_train, Ytr, X_test, Yte

    def load_CIFAR_batch(self, filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = self.load_pickle(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 1, 2, 3, ).astype("float32")

            Y = np.array(Y).astype("int64")
            return X, Y

    def load_pickle(self, f):
        version = platform.python_version_tuple()
        if version[0] == '2':
            return pickle.load(f)
        elif version[0] == '3':
            return pickle.load(f, encoding='latin1')
        raise ValueError("invalid python version: {}".format(version))

if __name__ == '__main__':
    g = GetDataSet("imagenette2")
    # print(g.train_data)
    # print(g.train_label)
# g = GetDataSet("EMNIST")
# print(g.train_data)
# print(g.train_label)
