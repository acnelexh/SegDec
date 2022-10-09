import numpy as np
import pickle
import os
from data.dataset import Dataset
from config import Config
from pathlib import Path
#import pdb

def read_split(kind: str):
    if kind == 'TRAIN':
        return train_samples
    elif kind == 'TEST':
        return test_samples
    elif kind == 'VAL':
        return validation_samples
    else:
        raise Exception('Unknown')



class JGPDataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(JGPDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        # Hard code
        self.image_dir = Path("./datasets/JGP/")
        self.cfg.ON_DEMAND_READ = True

        if kind == "TRAIN":
            self.image_dir = self.image_dir/"train"
        elif kind == "VAL":
            self.image_dir = self.image_dir/"val"
        elif kind == "TEST":
            self.image_dir = self.image_dir/"test"

        self.neg_files = [x for x in (self.image_dir/"neg").glob("*.png")]
        self.pos_files = (self.image_dir/"pos").glob("*GT.png")
        #self.pos_files = [x for x in self.pos_files][:len(self.neg_files)]
        # Read contents
        #pdb.set_trace()
        self.read_contents()

    def read_contents(self):
        pos_samples, neg_samples = [], []
        
        is_segmented = True

        for file in self.pos_files:
            sample = file.stem
            filename = str(file.stem)[:6]
            img_name = f"{filename}.png"
            img_path = str(self.image_dir/"pos"/img_name)
            seg_mask_path = str(self.image_dir/"pos"/f"{file.stem}.png")
            pos_samples.append((None, None, None, is_segmented, img_path, seg_mask_path, img_name))
            
        for file in self.neg_files:
            filename = str(file.stem)[:6]
            img_name = f"{filename}.png"
            img_path = str(self.image_dir/"neg"/img_name)
            #seg_mask_path = self.image_dir/"neg"/f"{sample}_GT.png"
            neg_samples.append((None, None, None, is_segmented, img_path, None, img_name))
                
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        self.len = 2 * len(pos_samples) if self.kind in ['TRAIN'] else len(pos_samples) + len(neg_samples)

        print(self.kind, self.num_pos, self.num_neg, self.len)

        self.init_extra()