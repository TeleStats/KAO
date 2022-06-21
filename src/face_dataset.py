import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torchvision import transforms

import my_transforms


class FaceDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        # Data init
        self.imgs = None
        self.data = None
        # Data transforms
        self.transform = transform
        # Variables init
        self._list_imgs_labels_()

    def __getitem__(self, index):
        img = Image.open(self.imgs[index], 'r')
        img = img.convert('RGB')

        # For now put the label as the image name
        label = self.data[index]

        sample = {
            'img': img,
            'label': label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.imgs)

    def _list_imgs_labels_(self):
        # List all the images and labels available
        self.imgs = []
        self.data = []

        img_suffixes = ['.jpeg', '.jpg', '.png']
        politicians_path = sorted([img for img in self.data_path.iterdir()])
        for pol_path in politicians_path:
            self.imgs += [img for img in pol_path.iterdir() if img.suffix in img_suffixes]
            self.data += [pol_path.stem for img in pol_path.iterdir() if img.suffix in img_suffixes]

    def collate_fn(self, batch):
        imgs_list = []
        labels_list = []

        for b in batch:
            imgs_list.append(b['img'])
            labels_list.append(b['label'])

        return imgs_list, labels_list


def main():
    transform = transforms.Compose([
        my_transforms.ResizeImgBbox((160, 160)),
        my_transforms.ToTensor()
    ])

    face_dataset = FaceDataset(data_path=sample_path, transform=transform)
    face_dataloader = DataLoader(face_dataset, batch_size=2, shuffle=False, collate_fn=face_dataset.collate_fn)

    for i, batch in enumerate(face_dataloader):
        #Test for single batch
        if i == 2:
            break
        imgs, labels = batch[0], batch[1]
        for img, label in zip(imgs, labels):
            img = transforms.ToPILImage()(img).convert('RGB')
            img.show()


if __name__ == "__main__":
    sample_path = Path('/home/agirbau/work/politics/data/faces_politicians')
    main()
