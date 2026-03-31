import os
import numpy as np
import scipy.io as sio
import scipy.ndimage as ndimage
from PIL import Image

def precompute_density_maps(img_dir, gt_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    print(f"Starting precomputation for {len(img_names)} images...")
    
    for i, img_name in enumerate(img_names):
        img_path = os.path.join(img_dir, img_name)
        gt_name = 'GT_' + img_name.replace('.jpg', '.mat')
        gt_path = os.path.join(gt_dir, gt_name)
        out_path = os.path.join(out_dir, img_name.replace('.jpg', '.npy'))

        with Image.open(img_path) as img:
            w, h = img.size
        mat = sio.loadmat(gt_path)
        points = mat['image_info'][0,0][0,0][0]
        density_map = np.zeros((h, w), dtype=np.float32)
        for p in points:
            x, y = int(p[0]), int(p[1])
            if y < h and x < w:
                density_map[y, x] = 1.0
                
        density_map = ndimage.gaussian_filter(density_map, sigma=15, mode='constant')
        np.save(out_path, density_map)
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(img_names)} images...")
