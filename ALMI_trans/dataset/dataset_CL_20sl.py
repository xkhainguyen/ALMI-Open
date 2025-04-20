import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
from dataclasses import dataclass

@dataclass
class SegmentId:
    idx: int
    start: int
    stop: int

class BatchSampler(torch.utils.data.Sampler):
    def __init__(self,
                 dataset,
                 seq_len: int,
                 batch_size: int,
                 padding_len: int,
                 can_sample_beyong_end: bool = False):
        super().__init__()
        self.dataset = dataset
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.padding_len = padding_len
        self.can_sample_beyong_end = can_sample_beyong_end

        self._generate_segment_ids()

    def __len__(self):
        return len(self.indices)
    
    def __iter__(self):
        import random
        random.shuffle(self.indices)

        return iter(self.indices)

    def _generate_segment_ids(self,):
        self.indices = []

        for idx, name in enumerate(self.dataset.name_list):
            data = self.dataset.data_dict[name]
            motion_len = len(data['obs_actions'])
            if motion_len >= self.seq_len:
                if self.can_sample_beyong_end:
                    num_segments = motion_len + self.seq_len - 1
                else:
                    num_segments = motion_len - self.seq_len + 1 + self.padding_len
                
                for i in range(num_segments):
                    if self.can_sample_beyong_end:
                        start_idx = i - self.seq_len + 1
                        end_idx = start_idx + self.seq_len
                        self.indices.append(
                            SegmentId(idx, start_idx, end_idx)
                        )
                    else:
                        start_idx = i - self.padding_len
                        end_idx = start_idx + self.seq_len
                        self.indices.append(
                            SegmentId(idx, start_idx, end_idx)
                        )
                else:
                    ValueError

class ALMI_CL_20slDataset(data.Dataset):
    def __init__(self, dataset_name):     
        self.dataset_name = dataset_name

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
                obs_actions = np.load(pjoin(self.action_dir, '%s.npy'%name), allow_pickle=True).item()['obs']
                obs_actions = obs_actions.astype(np.float32)
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
                    data_dict[name] = {'obs_actions': obs_actions,
                                       'text':text_data}
                    new_name_list.append(name)
            except:
                print("wrong name:", name)
        self.data_dict = data_dict  
        self.name_list = new_name_list 
        print(f"data set len {len(self.data_dict)}")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, segment_id: SegmentId):
        data = self.data_dict[self.name_list[segment_id.idx]]
        
        obs_actions, text_data = data['obs_actions'], data['text']
        caption = text_data[0]['caption']
        
        obs_actions = obs_actions[segment_id.start:segment_id.stop]

        return caption, obs_actions
