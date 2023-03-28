from torch.utils.data import DataLoader
import torch
import data_utils.MFLoader_correct as mf
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models.detection as models

train_data = mf.MFLoader(2,'train')
image, target = train_data.dataset.__getitem__(3)

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
image2 = output[0]['masks'][0].view(256,256).detach().numpy() + output[0]['masks'][1].view(256,256).detach().numpy() + output[0]['masks'][3].view(256,256).detach().numpy() + output[0]['masks'][4].view(256,256).detach().numpy() + output[0]['masks'][5].view(256,256).detach().numpy()
plt.imshow(image2)

# plt.imshow(sample["input_map"][:,:,1])

plt.show()
