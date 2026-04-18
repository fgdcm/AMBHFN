import numpy as np
from torch.utils.data import Dataset
import xarray as xr
from pathlib import Path
import torch


def prepare_inputs_targets(len_time, input_gap, input_length, pred_shift, pred_length, samples_gap):
    assert pred_shift >= pred_length
    input_span = input_gap * (input_length - 1) + 1
    pred_gap = pred_shift // pred_length
    input_ind = np.arange(0, input_span, input_gap)
    target_ind = np.arange(0, pred_shift, pred_gap) + input_span + pred_gap - 1
    ind = np.concatenate([input_ind, target_ind]).reshape(1, input_length + pred_length)
    max_n_sample = len_time - (input_span+pred_shift-1)
    ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones((1, input_length+pred_length), dtype=int)
    return ind[::samples_gap]
   
class data_process(Dataset):
    def __init__(self, ossit, samples_gap):
        super().__init__()
        
        # Handle tuple input (uv, zt)
        if isinstance(ossit, tuple) and len(ossit) == 2:
            uv, zt = ossit
            self.ossit = np.concatenate((uv, zt), axis=1)
        else:
            self.ossit = ossit
        
        if self.ossit is not None:
            assert len(self.ossit.shape) == 4
            self.idx_uv = prepare_inputs_targets(self.ossit.shape[0], input_gap=1, input_length=24, pred_shift=24, pred_length=24, samples_gap=samples_gap)
            self.length = len(self.idx_uv)
        else:
            self.idx_uv = None
            self.length = 0

    def GetDataShape(self):
        if self.ossit is not None:
            return {'uv': (len(self.idx_uv), 48, self.ossit.shape[1], self.ossit.shape[2], self.ossit.shape[3])}
        else:
            return {'uv': (0, 0, 0, 0, 0)}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.ossit is not None:
            return self.ossit[self.idx_uv[idx]]
        else:
            return None