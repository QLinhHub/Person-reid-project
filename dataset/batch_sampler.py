from torch.utils.data.sampler import Sampler
import random


class BatchSampler(Sampler):
    '''
    Sampler used in Pytorch DataLoader.
    The core idea is to form batches by randomly sampling P classes (person indentities), and then randomly sampling K images of each class (person)
    , thus result in a batch of PK images.
    '''
    def __init__(self, dataset, num_classes, n_img_per_class, *args, **kwargs) -> None:
        super().__init__(dataset, *args, **kwargs)
        self.num_classes = num_classes
        self.n_img_per_class = n_img_per_class
        self.dataset = dataset
        self.batch_size = self.num_classes * self.n_img_per_class
        self.labels_unique = dataset.labels_unique
        self.labels_imgs_index_dict = dataset.labels_imgs_index_dict
        self.num_iters = len(self.labels_unique) // self.num_classes

    def __iter__(self):
        random.shuffle(self.labels_unique) # random shuffle labels (person identities)
        for key in self.labels_imgs_index_dict:
            random.shuffle(self.labels_imgs_index_dict[key]) # random shuffle images index belong a specific labels

        step = 0
        for _ in range(self.num_iters):
            label_batch = self.labels_unique[step:step+self.num_classes]
            step += self.num_classes
            idx = []

            # (len(self.labels_imgs_index_dict[lb]) max/lb: 72/139, min/lb: 2/84 with bounding_box_train dataset
            for lb in label_batch:
                if (len(self.labels_imgs_index_dict[lb]) > self.n_img_per_class):
                    rand_idx = random.sample(self.labels_imgs_index_dict[lb], self.n_img_per_class)
                else:
                    rand_idx = random.choices(self.labels_imgs_index_dict[lb], k=self.n_img_per_class)
                idx.extend(rand_idx)
            yield idx

    def __len__(self):
        return self.num_iters


if __name__ == '__main__':
    from dataset import Market1501
    from torch.utils.data import DataLoader
    
    ds = Market1501("data/Market-1501-v15.09.15/bounding_box_train")
    sampler = BatchSampler(ds, 18, 4)
    dl = DataLoader(ds, batch_sampler=sampler)
    # batch_data = next(iter(dl))
    # print(batch_data[0].shape)
    while True:
        ims, lbs, _ = next(iter(dl))
        print(lbs.shape)