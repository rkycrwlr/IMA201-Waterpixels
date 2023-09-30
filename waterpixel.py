#%%
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import cv2


#%%

img_path = 'BSDS300/images/train/176035.jpg'
np_img_grey_scale = np.array(ImageOps.grayscale(Image.open(img_path)))
np_img = np.array(Image.open(img_path))

grad_morphological = np.array(ndimage.morphological_gradient(np_img_grey_scale, size=(2,2)))

H,W = np_img_grey_scale.shape
print(H,W)

N_cells = 8
if H<W : H_cells, W_cells, sigma = N_cells, int(N_cells * W/H) + 1, int(H / (N_cells))
else   : H_cells, W_cells, sigma = int(N_cells * H/W) + 1, N_cells, int(W / (N_cells))

markers = []
for i in range(H_cells):
    for j in range(W_cells):
       markers.append([int(sigma/2) + sigma * i, int(sigma/2) + sigma * j])

print(H_cells, W_cells, H_cells * W_cells)

#%%
minima_g = np.zeros((H,W), dtype=np.uint8)
min_g = grad_morphological.min()
min_positions = [i for i in range(H*W) if grad_morphological[i//W, i%W] == min_g]
for val in min_positions:
    minima_g[val//W, val%W] = 255
#minima_g[np.argmin(grad_morphological)] = 255


#%%
f, axarr = plt.subplots(1,3)
axarr[0].imshow(np_img, cmap='gray')
axarr[1].imshow(grad_morphological, cmap='gray')
axarr[2].imshow(minima_g, cmap='gray')
plt.imsave("out1.png", grad_morphological , cmap='gray')
plt.imsave("out2.png", minima_g, cmap='gray')

#%%
def compute_minima_g(g_img):
    H,W = g_img.shape
    minima_g = np.zeros((H,W), dtype=np.intc)
    min_g = g_img.min()
    min_positions = [i for i in range(H*W) if g_img[i//W, i%W] == min_g]
    for val in min_positions:
        minima_g[val//W, val%W] = 1
    return minima_g

def compute_cells(np_img, rho=2/3, N_cells=8):
    H,W = np_img.shape
    global_mask = np.ones((H,W), dtype=np.intc)
    if H<W : H_cells, W_cells, sigma = N_cells, int(N_cells * W/H) + 1, int(H / (N_cells))
    else   : H_cells, W_cells, sigma = int(N_cells * H/W) + 1, N_cells, int(W / (N_cells))
    cells = np.zeros((H_cells, W_cells, int(sigma*rho), int(sigma*rho)), dtype=np.uint8)
    for i in range(H_cells):
        for j in range(W_cells):
            global_mask[int(sigma/2) + i*sigma - int(sigma/2*rho) : int(sigma/2) + i*sigma + int(sigma/2*rho), int(sigma/2) + j*sigma - int(sigma/2*rho) : int(sigma/2) + j*sigma +int(sigma/2*rho)] = 0
            cells[i,j] = np_img[int(sigma/2) + i*sigma - int(sigma/2*rho) : int(sigma/2) + i*sigma + int(sigma/2*rho), int(sigma/2) + j*sigma - int(sigma/2*rho) : int(sigma/2) + j*sigma +int(sigma/2*rho)]
    return global_mask, cells

#%%

N_cells = 8
rho = 2/3
img_paths = ["BSDS300/images/test/{}".format(elt) for elt in os.listdir("BSDS300/images/test")]
for i, img_path in enumerate(img_paths):
    np_img_grey_scale = np.array(ImageOps.grayscale(Image.open(img_path)))
    g_img = np.array(ndimage.morphological_gradient(np_img_grey_scale, size=(2,2)))

    sigma = int(min(np_img_grey_scale.shape)/N_cells)

    minima_g = compute_minima_g(g_img)
    global_mask, cells = compute_cells(minima_g)
    minima_g_cut = np.vectorize(lambda x: max(x,0))(minima_g - global_mask)
    
    markers = np.zeros(np_img_grey_scale.shape, dtype=np.uint8)
    for k in range(len(cells)):
        for l in range(len(cells[0])):
            (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(cells[k,l])
            idx, maxarea = 0,0
            for j in range(1,totalLabels):
                area = values[j, cv2.CC_STAT_AREA]
                if area > maxarea:
                    idx = j
                    maxarea = area
            markers[int(sigma/2) + k*sigma - int(sigma/2*rho) : int(sigma/2) + k*sigma + int(sigma/2*rho), int(sigma/2) + l*sigma - int(sigma/2*rho) : int(sigma/2) + l*sigma +int(sigma/2*rho)] = (label_ids == idx) * (k * len(cells[0]+ l))
    
    markers_coords = [[] for i in range(len(cells)*len(cells[0]))]
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            if markers[i,j] > 0:
                markers_coords[markers[i,j]].append((i,j))

    ########## BEAUCOUP TROP LENT #############
    dQ = np.zeros(np_img_grey_scale.shape, dtype=np.float64)
    for i in range(np_img_grey_scale.shape[0]):
        for j in range(np_img_grey_scale.shape[1]):
            min_dist = np.inf
            for k in range(len(markers_coords)):
                for elt in markers_coords[k]:
                    min_dist = min(((elt[0] - i)**2 + (elt[1] - j)**2)**0.5, min_dist)
            dQ[i,j] = min_dist

    dQ = 2/sigma * dQ
    plt.imsave("out_dQ.png", dQ, cmap="gray")


    # reg_grad


    # plt.imsave("out.png", markers, cmap="gray")
    # plt.imsave("out_b.png", minima_g_cut, cmap="gray")

    plt.imsave("gradients/{}.png".format(i), minima_g_cut, cmap="gray")