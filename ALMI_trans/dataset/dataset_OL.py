import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate

class ALMI_OL_Dataset(data.Dataset):
    def __init__(self, dataset_name, unit_length = 4, codebook_size = 1024, vq_name=None, use_tcn=True, max_motion_len=500):

        self.unit_length = unit_length
        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1
        if dataset_name == 'almi':
            self.data_root = './dataset/ALMI'
            self.text_dir = pjoin(self.data_root, 'texts')
            self.max_motion_length = max_motion_len // unit_length
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
                m_token_list = np.load(pjoin(self.data_root, vq_name, '%s.npy'%name))
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
                    data_dict[name] = {'m_token_list': m_token_list,
                                       'text':text_data}
                    new_name_list.append(name)
            except:
                pass
        self.data_dict = data_dict
        self.name_list = new_name_list
        print(f"data set len {len(self.data_dict)}")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_tokens, text_data = data['m_token_list'], data['text']
        caption = text_data[0]['caption']
        
        coin = np.random.choice([False, False, True])
        if coin:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]

        if m_tokens_len+1 < self.max_motion_length:
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
        else:
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        return caption, m_tokens.reshape(-1), m_tokens_len

def DATALoader(dataset_name,
                batch_size, 
                codebook_size, 
                vq_name, 
                unit_length=4,
                num_workers=8,
                use_tcn=True,
                max_motion_len=500) : 

    train_loader = torch.utils.data.DataLoader(ALMI_OL_Dataset(dataset_name, codebook_size = codebook_size, vq_name = vq_name, unit_length=unit_length, use_tcn=use_tcn, max_motion_len=max_motion_len),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last = True)
    

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


