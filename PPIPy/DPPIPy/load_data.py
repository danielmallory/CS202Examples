import pandas as pd
import torch

print('==> Loading Data')


def pair_crop_load(file, pos_weight, pNum):
    orig_data = pd.read_csv(file)
    crop_data = pd.DataFrame()

    count = 0
    for i in range(len(orig_data)):
        for j in range(pNum[orig_data[i][1]]):
            for k in range(pNum[ orig_data[i][2]]):
                crop_data[count][1] = orig_data[i][1]

                count += 1