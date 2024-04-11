import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


class TransductiveModel(nn.Module):
    def __init__(self, num_modalities, feature_dims, hidden_dim, output_dim):
        super(TransductiveModel, self).__init__()

        self.num_modalities = num_modalities
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.shared_space_matrix = nn.Parameter(torch.randn(sum(feature_dims), hidden_dim))
        self.popularity_prediction = nn.Linear(hidden_dim, output_dim)

    def forward(self, modalities):
        concatenated_modalities = torch.cat(modalities, dim=1)

        shared_space_representation = torch.matmul(concatenated_modalities, self.shared_space_matrix)

        popularity_scores = self.popularity_prediction(shared_space_representation)

        return popularity_scores


def laplacian_penalty(L, lambda_disagreement):
    penalty = 0
    for i in range(1, len(L)):
        penalty += torch.norm(L[0] - L[i][:L[0].shape[0], :L[0].shape[1]])

    return lambda_disagreement * penalty
