from torchvision import transforms
from torch.utils.data import Dataset
import glob
import PIL


class Market1501(Dataset):
    '''
    a wrapper of Market1501 dataset
    '''
    def __init__(self, data_path, train=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.train = train
        self.imgs_file = glob.glob(data_path + "/*.jpg")
        self.labels = [int(f.split("/")[-1].split("_")[0]) for f in self.imgs_file]  # person identity
        self.cams = [int(f.split("/")[-1].split("_")[1][1]) for f in self.imgs_file] # camera where person appear

        # apply data augmentation only when training
        if train:
            self.transforms = transforms.Compose([
                transforms.Resize((288, 144)),
                transforms.RandomCrop((256, 128)),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(), 
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
            ])
        
        self.labels_imgs_index_dict = dict()  # dict contain label (unique) as key and list of images index belong this label as value: {label: list of img index}
        self.labels_unique = list(set(self.labels))  # list contain unique label (unique identity)
        for label in self.labels_unique:
            label_img_index = [idx for idx, lb in enumerate(self.labels) if lb==label]
            self.labels_imgs_index_dict.update({label: label_img_index})

    def __len__(self):
        return len(self.imgs_file)

    def __getitem__(self, idx):
        img = PIL.Image.open(self.imgs_file[idx])  # torchvision transform module can only process PIL image
        img = self.transforms(img)
        return img, self.labels[idx], self.cams[idx]


if __name__ == '__main__':
    import cv2
    import numpy as np
    
    dataset = Market1501("data/Market-1501-v15.09.15/bounding_box_train")
    im, identity, cam = dataset[10]
    print(im.shape)
    print(im.min())
    print(im.max())
    print(identity) # label
    print(cam)
    im = im.permute(1,2,0)
    cv2.imshow("ss", np.array(im))
    cv2.waitKey(0)
    cv2.destroyAllWindows()