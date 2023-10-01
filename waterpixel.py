#%%
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
import os
import cv2
import math


#%%

img_path = 'BSDS300/images/train/176035.jpg'
im = Image.open(img_path)
im = im.resize((400,300))
np_img = np.array(im)
CELL_RADIUS = 20
RHO=2/3
#%%
print(np_img.shape)

#%%

def morphological_grad(img, k_val=3, gauss_sigma=2):
    if img.shape[-1] >= 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = ndimage.gaussian_filter(img,gauss_sigma)
    grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, np.ones((k_val, k_val)))
    grad = grad / grad.max()
    return grad

#%%
g_img = np.array(morphological_grad(np_img))
plt.imshow(g_img, cmap='gray')
plt.show()
#%%

class Cell():
    def __init__(self,x,y,r,rho,img):
        # Should be gradient image with uint8 values here
        if img.max() <= 1:
            img = (img*255).astype(np.uint8)
        self.img = img
        self.center = (x,y)
        self.corner = (
            self.constrain(int(y-r),0,self.img.shape[0]-1),
            self.constrain(int(x-r*np.sqrt(3)/2),0,self.img.shape[1]-1),
            self.constrain(int(y+r),0,self.img.shape[0]-1),
            self.constrain(int(x+r*np.sqrt(3)/2),0,self.img.shape[1]-1))
        self.rad = r
        self.rho = rho
        self.values, self.mask = self.get_values()
        self.marker = self.get_marker()
        
    def constrain(self,val,min_val, max_val):
        return max(min_val, min(val,max_val))

    def get_values(self):
        msk = Image.new("L", (self.img.shape[1] + 2*self.rad, self.img.shape[0] + 2*self.rad))
        msk_d = ImageDraw.Draw(msk)
        poly = [(self.rad + self.center[0] + math.cos(th) * self.rad * self.rho, 
                 self.rad + self.center[1] + math.sin(th) * self.rad * self.rho) 
                 for th in [math.pi/6 + i * 2 * math.pi / 6 for i in range(6)]]
        msk_d.polygon(poly, fill=1)
        msk = np.array(msk, dtype=np.uint8)
        msk = msk[self.rad:-self.rad, self.rad:-self.rad]
        values = self.img * msk
        return values, msk
    
    def get_marker(self):
        marker = np.zeros(self.img.shape)
        crop_img = self.img[self.corner[0]:self.corner[2], self.corner[1]:self.corner[3]]
        crop_mask = self.mask[self.corner[0]:self.corner[2], self.corner[1]:self.corner[3]]
        minima_img = np.zeros(crop_img.shape, dtype=np.uint8)
        minima_img[np.where(crop_img == crop_img.min())] = 1
        minima_crop = minima_img * crop_mask
        (totalLabels, label_ids, stats, centroid) = cv2.connectedComponentsWithStats(minima_crop)
        idx, max_area = 0, 0.0
        for j in range(1,totalLabels):
            area = stats[j, cv2.CC_STAT_AREA]
            if area > max_area:
                idx = j
                max_area = area
        if idx == 0:
            marker[self.constrain(int(self.center[1]),0,self.img.shape[0]-1), self.constrain(int(self.center[0]),0,self.img.shape[1]-1)] = 1
        else:
            indexes = np.where(label_ids==idx)
            marker[(indexes[0] + self.corner[0], indexes[1] + self.corner[1])] = 1
        return marker
        

plt.imshow(g_img, cmap='gray')
plt.show()
cell = Cell(100,200,50,1,g_img)
plt.imshow(cell.values, cmap='gray')
plt.show()
plt.imshow(cell.marker, cmap='gray')
plt.show()
#%%

def compute_cells(img, radius=CELL_RADIUS, rho=RHO):
    W,H = img.shape
    cells = []
    hex_wid = radius * np.sqrt(3)
    hex_hei = radius * 1.5
    for i in range(int(W/hex_hei)+1):
        for j in range(int(H/hex_wid)+1):
            center = (i*hex_hei, j*hex_wid) if i%2==0 else (i*hex_hei, j*hex_wid+hex_wid/2)
            cells.append(Cell(center[1], center[0], radius, rho, img))
    return cells

cells = compute_cells(g_img)
glob_mask = np.zeros(g_img.shape)
glob_marker = np.zeros(g_img.shape)
for cell in cells:
    glob_mask += cell.mask
    glob_marker += cell.marker
plt.imshow(glob_mask, cmap='gray')
plt.show()
plt.imshow(glob_marker, cmap='gray')
plt.show()

#%%

# N_cells = 8
# rho = 2/3
# img_paths = ["BSDS300/images/test/{}".format(elt) for elt in os.listdir("BSDS300/images/test")]
# for i, img_path in enumerate(img_paths):
#     np_img_grey_scale = np.array(ImageOps.grayscale(Image.open(img_path)))
#     g_img = np.array(ndimage.morphological_gradient(np_img_grey_scale, size=(2,2)))

#     sigma = int(min(np_img_grey_scale.shape)/N_cells)

#     minima_g = compute_minima_g(g_img)
#     global_mask, cells = compute_cells(minima_g)
#     minima_g_cut = np.vectorize(lambda x: max(x,0))(minima_g - global_mask)
    
#     markers = np.zeros(np_img_grey_scale.shape, dtype=np.uint8)
#     for k in range(len(cells)):
#         for l in range(len(cells[0])):
#             (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(cells[k,l])
#             idx, maxarea = 0,0
#             for j in range(1,totalLabels):
#                 area = values[j, cv2.CC_STAT_AREA]
#                 if area > maxarea:
#                     idx = j
#                     maxarea = area
#             markers[int(sigma/2) + k*sigma - int(sigma/2*rho) : int(sigma/2) + k*sigma + int(sigma/2*rho), int(sigma/2) + l*sigma - int(sigma/2*rho) : int(sigma/2) + l*sigma +int(sigma/2*rho)] = (label_ids == idx) * (k * len(cells[0]+ l))
    
#     markers_coords = [[] for i in range(len(cells)*len(cells[0]))]
#     for i in range(markers.shape[0]):
#         for j in range(markers.shape[1]):
#             if markers[i,j] > 0:
#                 markers_coords[markers[i,j]].append((i,j))

#     ########## BEAUCOUP TROP LENT #############
#     dQ = np.zeros(np_img_grey_scale.shape, dtype=np.float64)
#     for i in range(np_img_grey_scale.shape[0]):
#         for j in range(np_img_grey_scale.shape[1]):
#             min_dist = np.inf
#             for k in range(len(markers_coords)):
#                 for elt in markers_coords[k]:
#                     min_dist = min(((elt[0] - i)**2 + (elt[1] - j)**2)**0.5, min_dist)
#             dQ[i,j] = min_dist

#     dQ = 2/sigma * dQ
#     plt.imsave("out_dQ.png", dQ, cmap="gray")


#     # reg_grad


#     # plt.imsave("out.png", markers, cmap="gray")
#     # plt.imsave("out_b.png", minima_g_cut, cmap="gray")

#     plt.imsave("gradients/{}.png".format(i), minima_g_cut, cmap="gray")