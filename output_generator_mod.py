from torch.utils.data import DataLoader
import torch
import data_utils.MFLoader_correct as mf
import matplotlib.pyplot as plt
import torchvision.models.detection as models
import numpy as np
from PIL import Image
import pickle

own_image = True
image_path = 'MonteFloor_Reproduction\house.png' #only used if own_image == True
invert_image = True

score_cut_off = 0.5

if own_image:
    # Load the image and resize it to 256x256
    img = Image.open(image_path)
    img = img.resize((256, 256))

    # Convert the image to black and white
    img = img.convert('L')

    # Load the image into a NumPy array
    arr = np.array(img)
    arr = (arr - np.max(arr)) / np.max(arr)

    if invert_image:
        arr = (arr*-1)+1

    image = arr

    imaget = torch.tensor(arr, dtype=torch.float32).view(1,1,256,256)

else:
    test_image = 4

    train_data = mf.MFLoader(1,'test')
    image, target = train_data.dataset.__getitem__(test_image)
    print(np.max(image))
    
    imaget = torch.tensor(image, dtype=torch.float32).view(1,1,256,256)

model = models.maskrcnn_resnet50_fpn()
model.load_state_dict(torch.load('MonteFloor_Reproduction\weights_maskrcnn'))
# FloorMaskRCNN_weights

# model.load_state_dict(torch.load('MonteFloor_Reproduction\FloorMaskRCNN_weights'))
model.eval()

output = model(imaget)
with open('out.pkl','wb') as file:
    pickle.dump(output,file)


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
