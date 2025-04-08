import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


class MemoryDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all images in memory"""

    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = Image.fromarray(self.images[index])
        x = self.transform(x)
        y = self.labels[index]
        return x, y


def get_data(trn_data, tst_data, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    taskcla = []
    if class_order is None:
        num_classes = len(np.unique(trn_data['y']))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1

    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}

    # ALL OR TRAIN
    filtering = np.isin(trn_data['y'], class_order)
    if filtering.sum() != len(trn_data['y']):
        trn_data['x'] = trn_data['x'][filtering]
        trn_data['y'] = np.array(trn_data['y'])[filtering]
    for this_image, this_label in zip(trn_data['x'], trn_data['y']):
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)
        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x'].append(this_image)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])

    # ALL OR TEST
    filtering = np.isin(tst_data['y'], class_order)
    if filtering.sum() != len(tst_data['y']):
        tst_data['x'] = tst_data['x'][filtering]
        tst_data['y'] = tst_data['y'][filtering]
    for this_image, this_label in zip(tst_data['x'], tst_data['y']):
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)
        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(this_image)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x'].append(data[tt]['trn']['x'][rnd_img[ii]])
                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])

    # convert them to numpy arrays
    for tt in data.keys():
        for split in ['trn', 'val', 'tst']:
            data[tt][split]['x'] = np.asarray(data[tt][split]['x'])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order

def get_perm(trn_data, tst_data, num_tasks, validation=0.0, seed=42):
    """Prepare data for Domain Incremental Learning (DIL) with Permuted MNIST."""
    np.random.seed(seed)
    random.seed(seed)
    
    data = {}
    taskcla = []
    class_order = list(range(10))  # Digits 0-9 remain the same across tasks
    
    # Flatten images to apply permutations
    trn_x_flat = trn_data['x'].reshape(-1, 28*28)
    tst_x_flat = tst_data['x'].reshape(-1, 28*28)

    # Generate a different pixel permutation for each task
    permutations = [np.random.permutation(28*28) for _ in range(num_tasks)]
    
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = f'task-{tt}'
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}
        
        # Apply permutation to training and test data
        perm = permutations[tt]
        trn_x_permuted = trn_x_flat[:, perm].reshape(-1, 28, 28)
        tst_x_permuted = tst_x_flat[:, perm].reshape(-1, 28, 28)
        
        data[tt]['trn']['x'] = trn_x_permuted
        data[tt]['trn']['y'] = trn_data['y']
        data[tt]['tst']['x'] = tst_x_permuted
        data[tt]['tst']['y'] = tst_data['y']
        
        # Validation split
        if validation > 0.0:
            val_size = int(validation * len(trn_data['y']))
            val_indices = np.random.choice(len(trn_data['y']), val_size, replace=False)
            data[tt]['val']['x'] = data[tt]['trn']['x'][val_indices]
            data[tt]['val']['y'] = data[tt]['trn']['y'][val_indices]
            mask = np.ones(len(trn_data['y']), dtype=bool)
            mask[val_indices] = False
            data[tt]['trn']['x'] = data[tt]['trn']['x'][mask]
            data[tt]['trn']['y'] = data[tt]['trn']['y'][mask]
        
        # Store number of classes (always 10 for MNIST)
        data[tt]['ncla'] = 10
        taskcla.append((tt, 10))
    
    data['ncla'] = 10  # Total number of classes remains 10 (digits 0-9)
    
    return data, taskcla, class_order