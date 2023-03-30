from torch.utils.data import DataLoader
import torch
import data_utils.MFLoader_correct as mf
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models.detection as models

def ewma(prev,curr,beta=0.9):
   return beta*prev + (1-beta)*curr

train_data = mf.MFLoader(1,'train')

n_epochs = 50

model = models.maskrcnn_resnet50_fpn()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
model.train()

dataloader = train_data.data

i = -1
for epoch in range(n_epochs):
   for images, targets in dataloader:
      # Forward pass
      loss_dict = model(images, targets)
      losses = sum(loss for loss in loss_dict.values())
      
      # Backward pass
      optimizer.zero_grad()
      losses.backward()
      optimizer.step()

      if i == -1:
         loss = ewma(losses.item(),losses.item())
      else:
         loss = ewma(loss,losses.item())
      
      if i % 50 == 0:
         print(f'{100*i/3000.}% of epoch {epoch+1}/{n_epochs} completed, loss: {loss}')
         torch.save(model.state_dict(), 'weights_maskrcnn')
      
      i += 1


