import numpy as np
import cv2 as cv2
###test image
img=cv2.imread('dataset/train/Keanu Reeves/keanu_1.jpg')

### splitting b,g,r channels
b,g,r=cv2.split(img)

### getting differences between (b,g), (r,g), (b,r) channel pixels
r_g=np.count_nonzero(abs(r-g))
r_b=np.count_nonzero(abs(r-b))
g_b=np.count_nonzero(abs(g-b))

### sum of differences
diff_sum=float(r_g+r_b+g_b)

### finding ratio of diff_sum with respect to size of image
ratio=diff_sum/img.size

if ratio>0.005:
    print("image is color")
else:
    print("image is greyscale")