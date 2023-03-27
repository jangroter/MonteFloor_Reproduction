from torch.utils.data import DataLoader
import data_utils.MFLoader as mf
import numpy as np

train_data = mf.MFLoader(32,'train')
sample = train_data.dataset.__getitem__(3)

print(sample['polygons_list'])
