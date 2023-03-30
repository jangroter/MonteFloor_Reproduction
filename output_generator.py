from torch.utils.data import DataLoader
import torch
import data_utils.MFLoader_correct as mf
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models.detection as models

score_cut_off = 0.9
test_image = 923

train_data = mf.MFLoader(1,'train')
image, target = train_data.dataset.__getitem__(test_image)

model = models.maskrcnn_resnet50_fpn()
model.load_state_dict(torch.load('weights_maskrcnn'))

model.eval()

imaget = torch.tensor(image, dtype=torch.float32).view(1,1,256,256)
output = model(imaget)

print(output)


plt.figure()
plt.subplot(121)
plt.imshow(image*-1)
plt.subplot(122)
print(len(output[0]['labels']))

masks = np.zeros_like(output[0]['masks'][0].view(256,256).detach().numpy())
for i in range(len(output[0]['labels'])):
    if output[0]['scores'][i] > score_cut_off:
        masks += output[0]['masks'][i].view(256,256).detach().numpy()

plt.imshow(masks)
plt.show()
