import torch
import numpy as np
from pathlib import Path
import cv2 as cv
#print(torch.cuda.is_available())

data_path = Path("./datasets/JGP/pos")

width = []
height = []

#for file in data_path.glob("*.png"):
#   img = cv.imread(str(file))
#    height.append(img.shape[0])
#    width.append(img.shape[1])

# pos 72 x 66
# neg 109 x 89
#print(sum(height)/len(height)) 
#print(sum(width)/len(width)) 

for file in data_path.glob("*GT.png"):
    img = cv.imread(str(file))
    if np.max(img) != 255 or np.min(img) != 0:
        print(np.max(img), np.min(img))
        cv.imwrite("test.png", img)
    