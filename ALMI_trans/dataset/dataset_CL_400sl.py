import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

class ALMI_CL_400slDataset(data.Dataset):
    def __init__(self, dataset_name, max_motion_len=500):     
        self.dataset_name = dataset_name
        min_motion_len = 40
        self.max_motion_len = max_motion_len

        if dataset_name == 'almi':
            self.data_root = './dataset/ALMI'
            self.text_dir = pjoin(self.data_root, 'texts')
            self.action_dir = pjoin(self.data_root, 'actions')
        else:
            NotImplementedError
        
        split_file = pjoin(self.data_root, 'train_ALMI.txt')

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                # Read text
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag
                            if 'wave' not in caption:
                                continue
                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                print(f"flag wrong {name}, text {line}")
                        except:
                            pass

                if flag:
                    obs_actions = np.load(pjoin(self.action_dir, '%s.npy'%name), allow_pickle=True).item()['obs']
                    obs_actions = obs_actions.astype(np.float32)
                    if (len(obs_actions)) < min_motion_len or (len(obs_actions) >= max_motion_len):
                        continue
                    data_dict[name] = {'obs_actions': obs_actions,
                                       'length': len(obs_actions),
                                       'text':text_data}
                    new_name_list.append(name)
            except:
                print("wrong name:", name)
        
        self.data_dict = data_dict
        self.name_list = new_name_list
        print(f"data set len {len(self.data_dict)}")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        
        obs_actions, text_data, m_length = data['obs_actions'], data['text'], data["length"]
        caption = text_data[0]['caption']

        if m_length < self.max_motion_len:
            obs_actions = np.pad(obs_actions, ((0, self.max_motion_len-m_length), (0, 0)), mode='constant', constant_values=0)
        
        return caption, obs_actions, m_length

def DATALoader(dataset_name,
                batch_size, 
                max_motion_len,
                num_workers = 8) : 

    train_loader = torch.utils.data.DataLoader(ALMI_CL_400slDataset(dataset_name, max_motion_len),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last = True)

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


