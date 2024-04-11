from tqdm import tqdm

from model import TransductiveModel, laplacian_penalty
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MyData, custom_collate_fn


def main():
    train_data = MyData(r'train.pkl')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, collate_fn=custom_collate_fn)

    K = 2

    Z = [768, 768]

    model = TransductiveModel(num_modalities=K, feature_dims=Z, hidden_dim=256, output_dim=1)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 100

    lambda_disagreement = 1.0

    model.train()

    model = model.to('cuda')

    for epoch in tqdm(range(num_epochs)):

        for batch in train_loader:
            batch = [item.to('cuda') if isinstance(item, torch.Tensor) else item for item in batch]

            visual_feature_embedding, textual_feature_embedding, visual_feature_embedding_test, textual_feature_embedding_test, label = batch

            visual_feature_embedding = torch.cat([visual_feature_embedding, visual_feature_embedding_test], dim=0)

            textual_feature_embedding = torch.cat([textual_feature_embedding, textual_feature_embedding_test], dim=0)

            output = model.forward([visual_feature_embedding, textual_feature_embedding])

            loss_popularity = nn.MSELoss()(output[:len(label)], label)

            L = [torch.mm(X.T, X) for X in [visual_feature_embedding, textual_feature_embedding]]
            loss_disagreement = laplacian_penalty(L, lambda_disagreement)

            total_loss = loss_popularity + loss_disagreement

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        torch.save(model, f"./model_{epoch}.pth")


if __name__ == '__main__':
    main()
