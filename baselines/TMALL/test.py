import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

import torch

from dataset_test import MyData, custom_collate_fn

test_data = MyData(r'test.pkl')

test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, collate_fn=custom_collate_fn)

model = torch.load(r"")

model.eval()

total_test_step = 0

total_MAE = 0

total_nMSE = 0

total_SRC = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Testing'):
        batch = [item.to('cuda') if isinstance(item, torch.Tensor) else item for item in batch]

        visual_feature_embedding, textual_feature_embedding, label = batch

        output = model.forward([visual_feature_embedding, textual_feature_embedding])

        output = output.to('cpu')

        label = label.to('cpu')

        output = np.array(output)

        label = np.array(label)

        MAE = mean_absolute_error(label, output)

        SRC, _ = spearmanr(output, label)

        nMSE = np.mean(np.square(output - label)) / (label.std() ** 2)

        total_test_step += 1

        total_MAE += MAE

        total_SRC += SRC

        total_nMSE += nMSE

print('MAE: ', total_MAE / total_test_step)

print('nMSE: ', total_nMSE / total_test_step)

print('SRC: ', total_SRC / total_test_step)
