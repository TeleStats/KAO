import copy
from torchvision import transforms


class ResizeImgBbox:
    def __init__(self, new_size):
        self.new_size = new_size
        self.resize = transforms.Resize(new_size)  # height, width
        # self.new_size = new_size  # width, height

    def __call__(self, sample):
        sample_resized = copy.copy(sample)
        image = sample['img']
        image_resized = self.resize(image)
        sample_resized['img'] = image_resized

        return sample_resized


class ToTensor:
    def __init__(self):
        # __magic_method__
        self.ToTensor = transforms.ToTensor()

    def __call__(self, sample):
        sample_tensor = copy.copy(sample)
        sample_tensor['img'] = self.ToTensor(sample['img'])

        return sample_tensor
