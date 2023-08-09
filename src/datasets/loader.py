import numpy as np
import cv2
from torch import from_numpy
from torchvision.transforms import transforms, Compose
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset, Dataset

colormap = [[0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7],
            [8, 8, 8],
            [9, 9, 9],
            [10, 10, 10],
            [11, 11, 11],
            [12, 12, 12],
            [13, 13, 13],
            [14, 14, 14],
            [15, 15, 15],
            [16, 16, 16],
            [17, 17, 17],
            [18, 18, 18],
            [19, 19, 19],
            [20, 20, 20],
            [21, 21, 21],
            [22, 22, 22],
            [23, 23, 23],
            [24, 24, 24],
            [25, 25, 25],
            [26, 26, 26],
            [27, 27, 27],
            [28, 28, 28],
            [29, 29, 29],
            [30, 30, 30],
            [31, 31, 31],
            [32, 32, 32],
            [33, 33, 33],
            [34, 34, 34],
            [35, 35, 35],
            [36, 36, 36],
            [37, 37, 37],
            [38, 38, 38],
            [39, 39, 39],
            [40, 40, 40]]

class NYUv2(Dataset):

    def __init__(self, train=True, data_root='../../datasets/nyuv2'):
        self.train = train
        self.data_root = data_root
        self.rbg, self.depth, self.labels = self._read_image_path(root=self.data_root)
        
    def __getitem__(self, index):
        rbg = self.rbg[index]
        depth = self.depth[index]
        label = self.labels[index]

        rbg = cv2.imread(rbg)
        depth = cv2.imread(depth)
        label = cv2.imread(label)
        rbg, depth, label = self._img_transforms(rbg, depth, label)
        return rbg, depth, label
    
    def __len__(self):
        return len(self.labels)

    def _read_image_path(self, root):
        if self.train:
            flag = 'train'
        else:
            flag = 'test'
        data = np.loadtxt(f'{root}/{flag}.txt', dtype=str)
        n = len(data)
        rgb, depth, label = [None]*n, [None]*n, [None]*n
        for i, frame in enumerate(data):
            rgb[i] = f'{root}/{flag}/rgb/{frame}.png'
            depth[i] = f'{root}/{flag}/depth/{frame}.png'
            label[i] = f'{root}/{flag}/labels_40/{frame}.png'
        return rgb, depth, label
    
    def _image2label(self, image, colormap):
        cm2lbl = np.zeros(256**3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0]*256+cm[1]*256+cm[2])] = i
        image = np.array(image, dtype="int64")
        ix = (image[:,:,0]*256+image[:,:,1]*256+image[:,:,2])
        image2 = cm2lbl[ix]
        return image2

    def _img_transforms(self, rbg, depth, label):
        transform = transforms.Compose([transforms.ToTensor()])
        rbg = transform(rbg)
        depth = transform(depth)
        label = from_numpy(self._image2label(label, colormap))
        return rbg, depth, label

class SUNRGBD(Dataset):

    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __add__(self, other: Dataset) -> ConcatDataset:
        return super().__add__(other)
    
    def __len__(self):
        pass

if __name__ == '__main__':

    train_loader = DataLoader(NYUv2(train=True), batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(NYUv2(train=True), batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    
    for step, (b_rgb, b_depth, b_y) in enumerate(train_loader):
        if step > 0:
            break

    print("b_rgb.shape:",b_rgb.shape)
    print("b_depth.shape:",b_depth.shape)
    print("b_y.shape:",b_y.shape)

    b_rbg_numpy = b_rgb.data.numpy()
    b_depth_numpy = b_depth.data.numpy()
    b_rbg_numpy = b_rbg_numpy.transpose(0,2,3,1)
    b_depth_numpy = b_depth_numpy.transpose(0,2,3,1)


    import matplotlib.pyplot as plt
    b_y_numpy = b_y.data.numpy()
    plt.imsave('../../tmp/brbg-train.png', b_rbg_numpy[1])
    plt.imsave('../../tmp/bdepth-train.png', b_depth_numpy[1])
    plt.imsave('../../tmp/by-train.png', b_y_numpy[1])

    for step, (b_rgb, b_depth, b_y) in enumerate(test_loader):
        if step > 0:
            break

    print("b_rgb.shape:",b_rgb.shape)
    print("b_depth.shape:",b_depth.shape)
    print("b_y.shape:",b_y.shape)

    b_rbg_numpy = b_rgb.data.numpy()
    b_depth_numpy = b_depth.data.numpy()
    b_rbg_numpy = b_rbg_numpy.transpose(0,2,3,1)
    b_depth_numpy = b_depth_numpy.transpose(0,2,3,1)


    import matplotlib.pyplot as plt
    b_y_numpy = b_y.data.numpy()
    plt.imsave('../../tmp/brbg-test.png', b_rbg_numpy[1])
    plt.imsave('../../tmp/bdepth-test.png', b_depth_numpy[1])
    plt.imsave('../../tmp/by-test.png', b_y_numpy[1])