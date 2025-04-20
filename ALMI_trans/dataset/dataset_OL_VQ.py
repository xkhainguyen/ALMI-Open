import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class SegmentId:
    idx: int
    start: int
    stop: int

class VQBatchSampler(torch.utils.data.Sampler):
    def __init__(self,
                 dataset,
                 padding_len: int,
                 can_sample_beyong_end: bool = False):
        super().__init__()
        self.dataset = dataset
        self.seq_len = dataset.window_size
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

        for idx, data in enumerate(self.dataset.data):
            motion_len = len(data)
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


class ALMIVQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 64):
        self.window_size = window_size
        if dataset_name == 'almi':
            self.data_root = './dataset/ALMI'
            self.text_dir = pjoin(self.data_root, 'texts')
            self.action_dir = pjoin(self.data_root, 'actions')
        else:
            NotImplementedError
        
        split_file = pjoin(self.data_root, 'train_ALMI.txt')

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

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
                    if obs_actions.shape[0] < self.window_size:
                        continue
                    self.lengths.append(obs_actions.shape[0] - self.window_size)
                    self.data.append(obs_actions)
            except:
                pass

        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def compute_sampling_prob(self) : 
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, segment_id: SegmentId):
        motion = self.data[segment_id.idx]
        
        motion = motion[segment_id.start:segment_id.stop]

        return motion
