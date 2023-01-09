import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import io

class CheXpertData(Dataset):
    def __init__(self, label_path, mode='train'):
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
             transforms.Grayscale(num_output_channels=3),
             transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])])
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]]
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                image_path = fields[0]
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        labels.append(self.dict[1].get(value))
                    elif index == 2 or index == 6 or index == 10:
                        labels.append(self.dict[0].get(value))
                labels = list(map(int, labels))
                self._image_paths.append(image_path)
                self._labels.append(labels)
                self._image_paths.append(image_path)
                self._labels.append(labels)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = io.imread(self._image_path[idx])
        image = self.transform(image)
        labels = np.array(self._labels[idx]).astype(np.float32)
        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'val':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))