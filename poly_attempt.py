import numpy as np
import matplotlib.pyplot as plt
import rdp
import pickle
import cv2
from scipy.spatial import ConvexHull
from shapely import Polygon
from shapely.geometry import Polygon
from descartes import PolygonPatch


with open('out.pkl','rb') as file:
    input = pickle.load(file)

for i in range(1000):
    try:
        output_mask = input[0]['masks'][i,0].detach().numpy() # make the masks into numpy arrays
        output_mask[(output_mask<1) & (output_mask>0.3)]=1
        # binary_mask = (output_mask>=0.5).astype(np.uint8)
        binary_mask=(output_mask).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        top_contour = max(contours,key=cv2.contourArea)
        # edges = cv2.Canny(binary_mask,100,200)

        # Try using convex hull
        # hull = ConvexHull(output_mask)
        # top_contour = Polygon(output_mask[hull.vertices])
        # patch = PolygonPatch(top_contour, fc='blue', ec='black', alpha=0.5)

        point_list = [tuple(point[0]) for point in top_contour]
        print(point_list)
        point_list_rdp = list(rdp.rdp(point_list,epsilon=1))
        point_list.append(point_list[0])
        point_list_rdp.append(point_list_rdp[0])

        #PLOT
        plottable_base = list(zip(*point_list)) # base polygon from cv2 contour
        plottable_rdp = list(zip(*point_list_rdp))

        # plt.plot(plottable_base[0],plottable_base[1])
        # plt.scatter(plottable_base[0],plottable_base[1])

        plt.plot(plottable_rdp[0],plottable_rdp[1])
        plt.scatter(plottable_rdp[0],plottable_rdp[1])
        masks = np.zeros_like(input[0]['masks'][0].view(256,256).detach().numpy())

        for i in range(len(input[0]['labels'])):
            if input[0]['scores'][i] > 0.9:
                masks += input[0]['masks'][i].view(256,256).detach().numpy()
    except IndexError:
        break
plt.imshow(masks)
plt.show()
