import os
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt


def RemovePixel(data, loss_rate):
    batch_size = data.shape[0]
    channel_size = data.shape[1]
    img_weight = data.shape[2]
    img_height = data.shape[3]

    for b in range(batch_size):
        for c in range(channel_size):
            img = data[b][c]
            sum = img_weight*img_height
            while sum <= (sum*loss_rate):
                n = img_weight
                random_row_index = random.randint(0, (img_height-1))
                loc_1 = random.randint(0,n-1)
                loc_2 = random.randint(0,n-1)
                len = np.abs(loc_1-loc_2)
                if loc_1>loc_2:
                    img[random_row_index][loc_2:loc_1] = 0
                else:
                    img[random_row_index][loc_1:loc_2] = 0
                sum -= len

    return data