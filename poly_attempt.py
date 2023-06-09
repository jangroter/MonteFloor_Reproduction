import numpy as np
import matplotlib.pyplot as plt
import rdp
import pickle
import cv2
from scipy.spatial import ConvexHull
from shapely import Polygon
from shapely.geometry import Polygon
from descartes import PolygonPatch


with open("out.pkl", "rb") as file:
    input = pickle.load(file)

for i in range(10):
# for i in range(np.shape(input[0]["masks"])[0]):
    try:
        output_mask = (
            input[0]["masks"][i, 0].detach().numpy()
        )  # make the masks into numpy arrays
        output_mask[(output_mask <= 1) & (output_mask > 0.3)] = 1  # binarize
        # output_mask[(output_mask != 1)] = 0  # all other vals to zero
        binary_mask = (output_mask).astype(np.uint8)  #
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )  # use OpenCV to find contours
        top_contour = max(contours, key=cv2.contourArea)

        point_list = [
            tuple(point[0]) for point in top_contour
        ]  # just pick the outermoust contour of max values
        print(point_list)
        point_list_rdp = list(
            rdp.rdp(point_list, epsilon=1.5)
        )  # Use Ramer-Douglas-Peucker algorithm to simplify the polygon. Epsilon value determines degree of simplification.
        point_list.append(point_list[0])
        point_list_rdp.append(point_list_rdp[0])

        # PLOT
        plottable_base = list(zip(*point_list))  # base polygon from cv2 contour
        plottable_rdp = list(zip(*point_list_rdp))

        plt.plot(plottable_rdp[0], plottable_rdp[1])
        plt.scatter(plottable_rdp[0], plottable_rdp[1])
        # masks = np.zeros_like(input[0]["masks"][0].view(256, 256).detach().numpy())

        # for i in range(len(input[0]["labels"])):
        #     if input[0]["scores"][i] > 0.9:
        #         masks += input[0]["masks"][i].view(256, 256).detach().numpy()
    except IndexError:
        break

# plots fixed to be higher contrast
score_cut_off=0.8
cmap = 'Greys'
masks = np.zeros_like(input[0]['masks'][0].view(256,256).detach().numpy())
count=0
for i in range(len(input[0]['labels'])):
    if input[0]['scores'][i] > score_cut_off:
        count+=1
        #cmap = list_of_cmaps[i%len(list_of_cmaps)]
        masks = input[0]['masks'][i].view(256,256).detach().numpy()
        masks = np.ma.masked_array(masks, masks < 0.1)
        masks[masks>0.1] = 0.5*(count/10 + 0.2)%12
        plt.imshow(masks,cmap=cmap,vmin=0,vmax=1,alpha=1)
# plt.imshow(masks)
plt.show()
