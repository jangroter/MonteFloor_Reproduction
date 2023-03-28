from torch.utils.data import DataLoader
import torch
import data_utils.MFLoader_correct as mf
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models.detection as models

train_data = mf.MFLoader(2,'train')
image, target = train_data.dataset.__getitem__(1)

# ensure that the masks have the correct shape
print(target['masks'].shape)

# plot one of the masks
# image2 = target['masks'][1,:,:].numpy()
# # check to see if the bounding box coordinates correspond with the mask
# print(target['boxes'][1])

# plt.figure()
# plt.imshow(image2)
# plt.show()

model = models.maskrcnn_resnet50_fpn()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
model.train()

dataloader = train_data.data

bestloss = 100
for images, targets in dataloader:
   # Forward pass
   loss_dict = model(images, targets)
   losses = sum(loss for loss in loss_dict.values())
   
   # Backward pass
   optimizer.zero_grad()
   losses.backward()
   optimizer.step()

   print('loss:', losses.item())
   if losses.item() < bestloss:  
      torch.save(model.state_dict(), 'weights_maskrcnn')
      bestloss = losses.item()

# for i in range(10):
#    images, targets = next(iter(train_data.data))
# #    images = list(image.to(device) for image in images)
# #    targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
   
#    optimizer.zero_grad()
#    loss_dict = model(images, targets)
#    losses = sum(loss for loss in loss_dict.values())
   
#    losses.backward()
#    optimizer.step()
   
#    print(i,'loss:', losses.item())
# #    if i%200==0:
#            torch.save(model.state_dict(), str(i)+".torch")
#            print("Save model to:",str(i)+".torch")