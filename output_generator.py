from torch.utils.data import DataLoader
import torch
import data_utils.MFLoader_correct as mf
import matplotlib.pyplot as plt
import matplotlib
import torchvision.models.detection as models
import numpy as np
from PIL import Image

own_image = False
image_path = '24_galleriaborghese.jpg' #only used if own_image == True
invert_image = False

score_cut_off = 0.8

if own_image:
    # Load the image and resize it to 256x256
    img = Image.open(image_path)
    img = img.resize((256, 256))

    # Convert the image to black and white
    img = img.convert('L')

    # Load the image into a NumPy array
    arr = np.array(img)
    arr = (arr - np.max(arr)) / np.max(arr)
    print(np.min(arr))
    print(np.max(arr))
    if invert_image:
        arr = (arr*-1)+1

    image = arr

    imaget = torch.tensor(arr, dtype=torch.float32).view(1,1,256,256)

else:
    test_image = 14

    train_data = mf.MFLoader(1,'test')
    image, target = train_data.dataset.__getitem__(test_image)
    print(np.max(image))

    if invert_image:
        image = (image*-1)+1
    
    imaget = torch.tensor(image, dtype=torch.float32).view(1,1,256,256)

model = models.maskrcnn_resnet50_fpn()
model.load_state_dict(torch.load('weights_maskrcnn'))

model.eval()

output = model(imaget)

print(output)

cmap = matplotlib.cm.get_cmap('Spectral')
cmap.set_bad(color='none', alpha=0)

list_of_cmaps = ['Purples','Blues','Greens','Oranges','Reds']

plt.figure()
plt.subplot(121)
plt.imshow(image,cmap='Greys')
plt.axis('off')
ax = plt.subplot(122)
print(len(output[0]['labels']))

cmap = 'Paired'

masks = np.zeros_like(output[0]['masks'][0].view(256,256).detach().numpy())
for i in range(len(output[0]['labels'])):
    if output[0]['scores'][i] > score_cut_off:
        #cmap = list_of_cmaps[i%len(list_of_cmaps)]
        masks = output[0]['masks'][i].view(256,256).detach().numpy()
        masks = np.ma.masked_array(masks, masks < 0.1)
        masks[masks>0.1] = i/len(output[0]['labels'])
        ax.imshow(masks,cmap=cmap,vmin=0,vmax=1,alpha=1)

plt.axis('off')
plt.show()
