import pickle
from pathlib import Path

import cv2
import numpy as np


preds_names = [file.name for file in Path('40_default').iterdir() if file.name.endswith('pickle')]


for name in preds_names:
    mean_pred = []
    for folder in ['40_default', '60_resize_aug', '60_rotate_train', '10_rotate_aug', '2_gpu_40']:
        with Path(folder, Path(name).stem + '.pickle').open('rb') as file:
            pred = pickle.load(file).squeeze()

        mean_pred.append(pred)

    pred = np.mean(np.array(mean_pred), axis=0)

    pred = ((pred > 0.7).astype(np.int32) * 255).astype(np.uint8)

    cv2.imwrite(str(Path('images', name.replace('.pickle', '.png'))), pred)
