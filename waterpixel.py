#%%
import numpy as np
import scipy
from scipy import ndimage
import skimage
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import cv2
import math
import pickle
import tqdm


#%%

img_path = 'BSR/BSDS500/data/images/val/3096.jpg'
im = Image.open(img_path)
# im = im.resize((400,300))
np_img = np.array(im)
CELL_RADIUS = 20
RHO=2/3
K=10
#%%
print(np_img.shape)

#%%

def morphological_grad(img, k_val=3, gauss_sigma=2):
    if img.ndim >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,0]
    img = ndimage.gaussian_filter(img,gauss_sigma)
    grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, np.ones((k_val, k_val)))
    grad = grad / grad.max()
    return grad

#%%
g_img = np.array(morphological_grad(np_img))
print(g_img.shape)
plt.imshow(g_img, cmap='gray')
plt.show()
#%%

class Cell():
    def __init__(self,x,y,r,rho,img):
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

    def get_center(self):
        return (self.constrain(int(self.center[1]),0,self.img.shape[0]-1), self.constrain(int(self.center[0]),0,self.img.shape[1]-1))

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
        marker = np.zeros(self.img.shape, dtype=np.uint8)
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
    H,W = img.shape
    cells = []
    hex_wid = radius * np.sqrt(3)
    hex_hei = radius * 1.5
    for i in range(int(H/hex_hei)+1):
        for j in range(int(W/hex_wid)+1):
            center = (i*hex_hei, j*hex_wid) if i%2==0 else (i*hex_hei, j*hex_wid+hex_wid/2)
            cells.append(Cell(center[1], center[0], radius, rho, img))
    return cells

def get_glob_marker_center(cells):
    glob_marker_center = np.zeros(cells[0].img.shape, dtype=np.uint16)
    for i,cell in enumerate(cells):
        glob_marker_center[cell.get_center()] = 1
    return glob_marker_center

def get_glob_marker_distinct(cells):
    glob_marker_distinct = np.zeros(cells[0].img.shape, dtype=np.uint16)
    for i,cell in enumerate(cells):
        glob_marker_distinct += cell.marker * (i+1)
    return glob_marker_distinct



cells = compute_cells(g_img)
glob_marker_center = get_glob_marker_center(cells)
glob_marker_distinct = get_glob_marker_distinct(cells)

plt.imshow(glob_marker_center, cmap='gray')
plt.show()
plt.imshow(glob_marker_distinct, cmap='gray')
plt.show()

#%%

def DF(a,b,i,j,dist_map):
    return 2*abs(a*dist_map[i+a,j+b,1]) + 2*abs(b*dist_map[i+a,j+b,2]) + a**2 + b**2

def DL_distance_map(markers):
    W,H = markers.shape
    dist_map = np.zeros((W,H,3))
    for i in range(W):
        for j in range(H):
            if markers[i,j] != 0:
                dist_map[i,j] = np.array([0,0,0])
            else:
                dist_map[i,j] = np.array([np.inf,0,0])

    V_minus = [(-1,-1),(-1,0),(-1,1),(0,-1)]
    for i in range(1,W):
        for j in range(1,H-1):
            if markers[i,j] != 0:
                continue
            a,b = min((dist_map[i+elt[0],j+elt[1],0] + DF(elt[0],elt[1],i,j,dist_map), elt) for elt in V_minus)[1]
            dist_map[i,j,1], dist_map[i,j,2] = dist_map[i+a,j+b,1]+a, dist_map[i+a,j+b,2]+b
            dist_map[i,j,0] = dist_map[i+a,j+b,0] + DF(a,b,i,j,dist_map)

    V_plus = [(1,1),(1,0),(1,-1),(0,1),(0,0)]
    for i in range(W-2,-2,-1):
        for j in range(H-2,-2,-1):
            if markers[i,j] != 0:
                continue
            a,b = min((dist_map[i+elt[0],j+elt[1],0] + DF(elt[0],elt[1],i,j,dist_map), elt) for elt in V_plus)[1]
            dist_map[i,j,1], dist_map[i,j,2] = dist_map[i+a,j+b,1]+a, dist_map[i+a,j+b,2]+b
            dist_map[i,j,0] = dist_map[i+a,j+b,0] + DF(a,b,i,j,dist_map)

    return dist_map[:,:,0]

dist_map = DL_distance_map(glob_marker_distinct)
print((2/CELL_RADIUS*np.sqrt(dist_map)).max())
plt.imshow(np.sqrt(dist_map))
plt.show()

#%%

def compute_reg_grad(g_img, dist_map, k=K, cell_dist=CELL_RADIUS):
    return g_img + k * (2/cell_dist) * np.sqrt(dist_map)

reg_g_img = compute_reg_grad(g_img*255, dist_map)

plt.imshow(reg_g_img, cmap='gray')
plt.show()

#%%

labels = skimage.segmentation.watershed(reg_g_img, glob_marker_distinct, watershed_line=True)
plt.imshow(labels)
plt.show()
#%%

def compute_segmentation(img, labels):
    waterpix_mask = (labels == 0) * 255
    if img.ndim >= 3:
        waterpix_mask = np.tile(np.expand_dims(waterpix_mask, axis=-1),3)
    waterpix_mask = skimage.morphology.dilation(waterpix_mask)
    return np.where(waterpix_mask, waterpix_mask, img)

def compute_segmentation_only_mask(labels):
    waterpix_mask = (labels == 0)
    waterpix_mask = waterpix_mask.astype(np.uint8)
    return waterpix_mask

img_gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
plt.imshow(compute_segmentation(img_gray, labels), cmap='gray')
plt.show()
waterpix_mask = compute_segmentation_only_mask(labels)
plt.imshow(waterpix_mask)
plt.show()
#%%
def waterpixel(img, cell_rad, k, rho, marker_center=False, only_labels=False, only_mask=False):
    g_img = morphological_grad(img)
    cells = compute_cells(g_img, radius=cell_rad, rho=rho)
    glob_marker_distinct = get_glob_marker_distinct(cells)

    if marker_center:
        glob_marker_center = get_glob_marker_center(cells)
        dist_map = DL_distance_map(glob_marker_center)
    else:
        dist_map = DL_distance_map(glob_marker_distinct)

    reg_g_img = compute_reg_grad(g_img*255, dist_map, k, cell_dist=CELL_RADIUS)

    labels = skimage.segmentation.watershed(reg_g_img, glob_marker_distinct, watershed_line=True)
    
    if only_labels:
        return labels
    
    if only_mask:
        waterpix_img = compute_segmentation_only_mask(labels)
    else :
        waterpix_img = compute_segmentation(img,labels)
    return waterpix_img

# img_gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
w_img = waterpixel(np_img, 10, 8, 2/3)
plt.imshow(w_img)
plt.show()

labels = waterpixel(np_img, 10, 8, 2/3, only_labels=True)
plt.imshow(labels)
plt.show()
# %%

def contourDensity(border):
    w,h=border.shape[:2]
    sb = 2*(w+h)-4
    sc = border.sum()
    d = w*h
    return (sc+sb)/d

#%%
def barycenters(labels):
    indiceMax = labels.max()
    barys = []
    for idx in range(1,indiceMax+1):
        coordsx,coordsy = np.where(labels==idx)
        baryX = coordsx.sum() / len(coordsx)
        baryY = coordsy.sum() / len(coordsy)
        barys.append((int(baryX.round()),int((baryY.round()))))
    return barys


def average_superpix(labels, barys):
    w,h = labels.shape
    avg = np.zeros((2*w,2*h))
    indiceMax = labels.max()
    
    for idx in range(1,indiceMax+1):
        x,y = np.where(labels==idx)
        coordsX = x - barys[idx-1][0] + w
        coordsY = y - barys[idx-1][1] + h
        for i in range(len(x)):
            avg[coordsX[i],coordsY[i]] += 1
    avg = avg/indiceMax
    avg = avg>=0.5

    return avg

barys = barycenters(labels)
avg = average_superpix(labels,barys)
plt.imshow(avg)
plt.show()

# %%
def centered_label(i, barys, labels):
    w,h = labels.shape
    result = np.zeros((2*w,2*h))
    x,y = np.where(labels==i)
    for j in range(len(x)):
        result[x[j]-barys[i-1][0]+w,y[j]-barys[i-1][1]+h] = 1
    return result



def mf(img1, img2):
    inter = np.logical_and(img1, img2).sum()
    union = np.logical_or(img1, img2).sum()
    return 1 - inter / union

#vérification de la fonction mf
result = mf(centered_label(1, barys, labels), centered_label(1, barys, labels))
print(result)

# %%

def missmatchFactor(labels):
    barys = barycenters(labels)
    avg = average_superpix(labels, barys)
    N = labels.max()
    S = 0
    for i in range(N):
        S += mf(centered_label(i,barys,labels), avg)
    return S/N


result = missmatchFactor(labels)
print(result)

# %%

gt_file = scipy.io.loadmat('BSR/BSDS500/data/groundTruth/val/3096.mat')
gt = np.array(gt_file['groundTruth'][0][0][0][0][1], dtype=np.float64)

plt.imshow(gt)
plt.show()

def L1_distance_map(markers):
    W,H = markers.shape
    dist_map = np.zeros((W,H))

    for i in range(W):
        for j in range(H):
            if markers[i,j] == 0:
                dist_map[i,j] = np.inf

    V_minus = [(-1,0),(0,-1),(0,0)]
    V_plus = [(1,0),(0,1),(0,0)]
    for i in range(1,W):
        for j in range(1,H):
            if markers[i,j] != 0:
                continue
            dist_map[i,j] = min([dist_map[i+elt[0],j+elt[1]] + 1 for elt in V_minus])
    
    for i in range(W-2,-2,-1):
        for j in range(H-2,-2,-1):
            if markers[i,j] != 0:
                continue
            dist_map[i,j] = min([dist_map[i+elt[0],j+elt[1]] + 1 for elt in V_plus])

    return dist_map

wat = waterpixel(np_img, 20, 25, 2/3, only_mask=True)
plt.imshow(wat)
plt.show()
d_map = L1_distance_map(wat)
plt.imshow(d_map)
plt.show()
#%%

def boundaryRecall(GT,wat_mask):
    dist_map = L1_distance_map(wat_mask)
    pos = np.where(GT == 1)
    S = np.sum(dist_map[pos] < 3)
    return S / np.sum(GT)

print(boundaryRecall(gt, wat))

#%%

def get_n_seg(rad,width,height):
    return int((width / (rad * np.sqrt(3))) * (height / (rad * 1.5)))

folder = 'BSR/BSDS500/data/'
images = sorted([folder + 'images/val/' + elt for elt in os.listdir(folder + 'images/val/')])
gt_names = sorted([folder + 'groundTruth/val/' + elt for elt in os.listdir(folder + 'groundTruth/val')])

print(len(images))

img = np.array(Image.open(images[0]))
h,w = img.shape[:2]

slic_labels = skimage.segmentation.slic(img, n_segments=get_n_seg(20,w,h))
slic_bounds = skimage.segmentation.find_boundaries(slic_labels)
wat_labels = w_img = waterpixel(img, 20, 8, 2/3, only_labels=True)

plt.imshow(img)
plt.show()

plt.imshow(slic_labels)
plt.title(str(slic_labels.max()))
plt.show()

plt.imshow(wat_labels)
plt.title(str(wat_labels.max()))
plt.show()

#%%


N_img = 20

k_values = [0,4,8,16]
rad_values = [10,20,30,40]

all_wat_results = np.zeros((4,4,3))
all_slic_results = np.zeros((4,3))

for img_name, gt_name in tqdm.tqdm(zip(images[:N_img],gt_names[:N_img])):
    print(img_name)
    img = np.array(Image.open(img_name))
    h,w = img.shape[:2]
    gt_file = scipy.io.loadmat(gt_name)
    gt = np.array(gt_file['groundTruth'][0][0][0][0][1], dtype=np.float64)

    for i, rad in enumerate(rad_values):
        for j, k in enumerate(k_values):
            wat_labels = waterpixel(img, rad, k, 2/3, only_labels=True)
            wat_border = compute_segmentation_only_mask(wat_labels)
            CD = contourDensity(wat_border)
            MF = missmatchFactor(wat_labels)
            BR = boundaryRecall(gt,wat_border)
            all_wat_results[i,j] += np.array([CD,MF,BR])
        
        slic_labels = skimage.segmentation.slic(img, n_segments=get_n_seg(rad,w,h))
        slic_bounds = skimage.segmentation.find_boundaries(slic_labels)
        CD = contourDensity(slic_bounds)
        MF = missmatchFactor(slic_labels)
        BR = boundaryRecall(gt,slic_bounds)
        all_slic_results[i] += np.array([CD,MF,BR])

results = {'wat': all_wat_results / N_img, 'slic': all_slic_results / N_img}
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)

#%%

with open('results.pkl', 'rb') as f:
    res = pickle.load(f)
    res_wat = res['wat']
    res_slic = res['slic']

#%%

wat_k0_BR = res_wat[:,0,2]
wat_k8_BR = res_wat[:,2,2]
wat_k16_BR = res_wat[:,3,2]
slic_BR = res_slic[:,2]
wat_k0_CD = res_wat[:,0,0]
wat_k8_CD = res_wat[:,2,0]
wat_k16_CD = res_wat[:,3,0]
slic_CD = res_slic[:,0]
wat_k0_MF = res_wat[:,0,1]
wat_k8_MF = res_wat[:,2,1]
wat_k16_MF = res_wat[:,3,1]
slic_MF = res_slic[:,1]

plt.plot(wat_k0_BR, wat_k0_CD, label="Waterpixel K=0")
plt.plot(wat_k8_BR, wat_k8_CD, label="Waterpixel K=8")
plt.plot(wat_k16_BR, wat_k16_CD, label="Waterpixel K=16")
plt.plot(slic_BR, slic_CD, label="Slic")

plt.title("Contour_Density_vs_Boundary_Recall")
plt.xlabel("Boundary recall")
plt.ylabel("Contour Density")
plt.legend()
plt.show()

plt.plot(rad_values, wat_k0_CD, label="Waterpixel K=0")
plt.plot(rad_values, wat_k8_CD, label="Waterpixel K=8")
plt.plot(rad_values, wat_k16_CD, label="Waterpixel K=16")
plt.plot(rad_values, slic_CD, label="Slic")

plt.title("Contour_Density_vs_Superpixel_Size")
plt.xlabel("Superpixel Size")
plt.ylabel("Contour Density")
plt.legend()
plt.show()

plt.plot(rad_values, wat_k0_MF, label="Waterpixel K=0")
plt.plot(rad_values, wat_k8_MF, label="Waterpixel K=8")
plt.plot(rad_values, wat_k16_MF, label="Waterpixel K=16")
plt.plot(rad_values, slic_MF, label="Slic")

plt.title("Mismatch_Factor_vs_Superpixel_Size")
plt.xlabel("Superpixel Size")
plt.ylabel("Mismatch Factor")
plt.legend()
plt.show()

plt.plot(rad_values, wat_k0_BR, label="Waterpixel K=0")
plt.plot(rad_values, wat_k8_BR, label="Waterpixel K=8")
plt.plot(rad_values, wat_k16_BR, label="Waterpixel K=16")
plt.plot(rad_values, slic_BR, label="Slic")

plt.title("Boundary_Recall_vs_Superpixel_Size")
plt.xlabel("Superpixel Size")
plt.ylabel("Boundary Recall")
plt.legend()
plt.show()

plt.plot(wat_k0_BR, wat_k0_MF, label="Waterpixel K=0")
plt.plot(wat_k8_BR, wat_k8_MF, label="Waterpixel K=8")
plt.plot(wat_k16_BR, wat_k16_MF, label="Waterpixel K=16")
plt.plot(slic_BR, slic_MF, label="Slic")

plt.title("Mismatch_Factor_vs_Boundary_Recall")
plt.xlabel("Boundary recall")
plt.ylabel("Mismatch Factor")
plt.legend()
plt.show()
