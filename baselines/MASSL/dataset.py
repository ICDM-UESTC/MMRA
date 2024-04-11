import torch.utils.data
import pandas as pd


def custom_collate_fn(batch):
    visual_feature_embedding, textual_feature_embedding, label = zip(*batch)

    return torch.tensor(visual_feature_embedding, dtype=torch.float), torch.tensor(textual_feature_embedding,
                                                                                   dtype=torch.float), \
        torch.tensor(label, dtype=torch.float).unsqueeze(1)


class MyData(torch.utils.data.Dataset):

    def __init__(self, path):
        super().__init__()

        self.path = path

        self.dataframe = pd.read_pickle(path)

        self.visual_feature_embedding_list = self.dataframe['visual_feature_embedding_cls'].tolist()

        self.textual_feature_embedding_list = self.dataframe['textual_feature_embedding'].tolist()

        self.label_list = self.dataframe['label'].tolist()

    def __getitem__(self, index):
        visual_feature_embedding = self.visual_feature_embedding_list[index]

        textual_feature_embedding = self.textual_feature_embedding_list[index]

        label = self.label_list[index]

        return visual_feature_embedding, textual_feature_embedding, label

    def __len__(self):
        return len(self.dataframe)

