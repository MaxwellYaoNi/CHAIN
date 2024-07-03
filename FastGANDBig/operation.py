import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image
from copy import deepcopy
import json
import time, random
import torchvision.transforms.functional as tvF
from typing import Any



def InfiniteSampler(n):
    """Data sampler"""
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def copy_G_params(model):
    with torch.no_grad():
        flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
    

def load_params(model, new_param):
    with torch.no_grad():
        for p, new_p in zip(model.parameters(), new_param):
            p.data.copy_(new_p)


class  ImageFolder(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, root, transform=None):
        super( ImageFolder, self).__init__()
        self.root = root

        self.frame = self._parse_frame()
        self.transform = transform

    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg' or image_path[-4:] == '.tif' :
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
            
        if self.transform:
            img = self.transform(img) 

        return img



from io import BytesIO
import lmdb
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            #key_asp = f'aspect_ratio-{str(index).zfill(5)}'.encode('utf-8')
            #aspect_ratio = float(txn.get(key_asp).decode())

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

class MetricsLogger(object):
    def __init__(self, fname, reinitialize=False):
        self.fname = fname
        self.reinitialize = reinitialize
        if os.path.exists(self.fname):
            if self.reinitialize:
                print('{} exists, deleting...'.format(self.fname))
                os.remove(self.fname)

    def log(self, record=None, **kwargs):
        """
        Assumption: no newlines in the input.
        """
        if record is None:
            record = {}
        record.update(kwargs)
        record['_stamp'] = time.time()
        with open(self.fname, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=True) + '\n')

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

def seed_rng(seed, torch_seed=3407):
    # torch_seed = seed
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MinSizeCenterCrop(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        min_dimension = min(img.size)
        return tvF.center_crop(img, min_dimension)


def get_dir(config):
    task_name = f'{config.dataset_name}'
    task_name += f'_seed{config.seed}' if config.seed else ''

    if config.chain_type:
        chain_name = f"{config.chain_type}_Blk{config.chain_blocks}{'' if config.chain_place == 'C1C2CS' else ('_P%s' % config.chain_place)}_tau{config.tau:g}_lbd{config.lbd:g}"
        chain_name += f'_lbdp0-{config.lbd_p0:g}' if config.lbd_p0 else ''
        task_name = f'{chain_name}-{task_name}'

    task_path = os.path.join(config.out_dir, task_name)
    saved_model_folder = os.path.join(task_path, 'models')
    saved_image_folder = os.path.join(task_path, 'images')
    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)
    config.metrics_log_path = os.path.join(config.out_dir, task_name + '_metrics.jsonl')

    return saved_model_folder, saved_image_folder
