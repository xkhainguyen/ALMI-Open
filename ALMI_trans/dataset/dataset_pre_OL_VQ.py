import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

class Pre_OL_VQDataset(data.Dataset):
    def __init__(self, dataset_name, unit_length = 8, max_motion_len=500):
        self.unit_length = unit_length

        self.dataset_name = dataset_name
        min_motion_len = 40
        
        if dataset_name == dataset_name == 'almi':
            self.data_root = './dataset/ALMI'
            self.text_dir = pjoin(self.data_root, 'texts')
            self.action_dir = pjoin(self.data_root, 'actions')
        else:
            NotImplementedError

        split_file = pjoin(self.data_root, 'train_ALMI.txt')
        
        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            skip = False
            try:
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    lines = f.readlines()
                    for line in lines:
                        try:
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            if 'wave' not in caption:
                                skip = True
                        except:
                            pass
                if not skip:
                    obs_actions = np.load(pjoin(self.action_dir, '%s.npy'%name), allow_pickle=True).item()['obs']
                    obs_actions = obs_actions.astype(np.float32)    
                    if (len(obs_actions)) < min_motion_len or (len(obs_actions) >= max_motion_len):
                        continue
                    data_dict[name] = {'motion': obs_actions,
                                    'length': len(obs_actions),
                                    'name': name}
                    new_name_list.append(name)
                    length_list.append(len(obs_actions))
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        return motion, name

def DATALoader(dataset_name,
                batch_size = 1,
                num_workers = 8, 
                unit_length = 4,
                max_motion_len = 500) : 
    
    train_loader = torch.utils.data.DataLoader(Pre_OL_VQDataset(dataset_name, unit_length=unit_length,max_motion_len=max_motion_len),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
