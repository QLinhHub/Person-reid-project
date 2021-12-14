import numpy as np
import torch

class BatchHardSelector(object):
    '''
    For each sample in the batch selected from batch_sampler.py, after forward propagation, we can select the hardest positive and 
    the hardest negative samples within the batch when forming the triplets FOR COMPUTING THE LOSS, which we call Batch Hard.
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def __call__(self, embeds, labels):
        dist_mat = torch.cdist(embeds, embeds).detach().cpu().numpy() # compute distance matrix: n x n
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1)) # convert labels to 2 demesions array, need this format to use with np.diag_indices
        labels_same = labels == labels.T # construct label matrix: n x n
        labels_diag_idx = np.diag_indices(labels_same.shape[0]) # get the diagonal index of the matrix

        # search for hardest positive embeds
        labels_pos = labels_same.copy()
        labels_pos[labels_diag_idx] = False 
        dist_pos_mat = dist_mat.copy()
        dist_pos_mat[labels_pos==False] = -np.inf
        pos_idx = np.argmax(dist_pos_mat, axis=1) # get the index of the hardest positives embeds: n
        pos_embeds = embeds[pos_idx]

        # search for hardest negative embeds
        labels_neg = labels_same.copy()
        dist_neg_mat = dist_mat.copy()
        dist_neg_mat[labels_neg==True] = np.inf
        neg_idx = np.argmin(dist_neg_mat, axis=1) # get the index of the hardest negatives embeds: n
        neg_embeds = embeds[neg_idx]

        return embeds, pos_embeds, neg_embeds


if __name__ == '__main__':
    embeds = torch.randn(10, 128)
    labels = torch.tensor([0,1,2,2,0,1,2,1,1,0])
    selector = BatchHardSelector()
    embeds, pos_embeds, neg_embeds = selector(embeds, labels)


