import numpy as np
import pickle
import os
from data.dataset import Dataset
from config import Config
from pathlib import Path
#import pdb
import pprint
from collections import defaultdict

pp = pprint.PrettyPrinter(indent=4)

def read_split(kind: str):
    if kind == 'TRAIN':
        return train_samples
    elif kind == 'TEST':
        return test_samples
    elif kind == 'VAL':
        return validation_samples
    else:
        raise Exception('Unknown')


class JGPMultiDataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(JGPMultiDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        # Hard code
        self.image_dir = Path("./datasets/JGP")
        self.cfg.ON_DEMAND_READ = True

        if kind == "TRAIN":
            self.image_dir = self.image_dir/"train"
        elif kind == "VAL":
            self.image_dir = self.image_dir/"val"
        elif kind == "TEST":
            self.image_dir = self.image_dir/"test"
        
        # add labels
        self.labels = [str(d).split("/")[-1] for d in self.image_dir.glob('*')]
        # assume negative label always on idx 0
        tmp = ["neg"]
        for l in self.labels:
            if l != "neg":
                tmp.append(l)
        self.labels = tmp

        self.labels_lookup = dict()
        # assume we created the each labels is presented in each split
        for idx, l in enumerate(self.labels):
            self.labels_lookup[l] = idx
        
        print(self.labels_lookup)

        # use dictionary of list to store files
        self.files = dict()
        for l in self.labels:
            all_files =  [x for x in (self.image_dir/l).glob("*")]
            self.files[l] =  [x for x in all_files if x.stem[-3:] != "_GT"]
        #pp.pprint(self.files)
        # Read contents
        self.read_contents()

    def read_contents(self):

        self.samples = []
        
        is_segmented = True

        for label_l in self.files:
            files_label_l = self.files[label_l]
            for f in files_label_l:
                img_name = f"{str(f.stem)}.png"
                img_path = str(self.image_dir/label_l/img_name)
                label_idx = self.labels_lookup[label_l]
                if label_l != "neg":
                    seg_mask_path = str(self.image_dir/label_l/f"{f.stem}_GT.png")
                    self.samples.append((None, None, None, is_segmented, img_path, seg_mask_path, img_name, label_idx))
                else:
                    self.samples.append((None, None, None, is_segmented, img_path, None, img_name, label_idx))
        
        '''
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        self.len = 2 * len(pos_samples) if self.kind in ['TRAIN'] else len(pos_samples) + len(neg_samples)
        '''
        self.len = len(self.samples)

        # hard code for now, subject to change
        #self.num_neg = len(samples["neg"])
        #self.num_pos = sum([len(v) for k, v in samples.items() if k != "neg"])
        #print(self.kind, self.len)
        #pp.pprint(samples['a'])
        #self.init_extra()