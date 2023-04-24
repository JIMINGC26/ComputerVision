import os
from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

CROP_SIZE = 32

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath)
    # y, _, _ = img.split()
    return img

class SRGANDataset(Dataset):
    def __init__(self, data_path, zoom_factor ,ty="train"):
        self.image_filenames = [join(data_path, x) for x in listdir(data_path) if is_image_file(x)]
        
        crop_size = CROP_SIZE - (CROP_SIZE % zoom_factor) # Valid crop size
        # self.tfs = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

        self.input_transform = T.Compose([T.CenterCrop(crop_size), # cropping the image
                                      T.Resize(crop_size//zoom_factor),  # subsampling the image (half size)
                                      T.Resize(crop_size, interpolation=Image.BICUBIC),  # bicubic upsampling to get back the original size 
                                      T.ToTensor(),
                                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.target_transform = T.Compose([T.CenterCrop(crop_size), # since it's the target, we keep its original quality
                                       T.ToTensor(),
                                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.dataset = []
        # self.path = data_path
        # self.ty = ty
        # f = open(os.path.join(data_path, "{}.txt".format(ty)))
        # self.dataset.extend(f.readlines())
        # f.close()
        # self.tfs = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        
        # input = input.filter(ImageFilter.GaussianBlur(1)) 
        input = self.input_transform(input)
        target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


if __name__ == '__main__':
    e = SRGANDataset("dataset/Set5", zoom_factor=2)
    a, b = e[0]
    print(a.shape, "\n", b.shape)


