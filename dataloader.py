from torch.utils.data import DataLoader
import data_utils.MFLoader_correct as mf
import numpy as np
import matplotlib.pyplot as plt

train_data = mf.MFLoader(32,'train')
image, target = train_data.dataset.__getitem__(1)

# ensure that the masks have the correct shape
print(target['masks'].shape)

# plot one of the masks
image2 = target['masks'][1,:,:].numpy()
# check to see if the bounding box coordinates correspond with the mask
print(target['boxes'][1])

plt.figure()
plt.imshow(image2)
plt.show()