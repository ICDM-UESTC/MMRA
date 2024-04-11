import torch.utils.data
import pandas as pd
from functools import partial


def custom_collate_fn(batch, num_of_retrieved_items, num_of_frames):

    visual_feature_embedding, textual_feature_embedding, similarity, retrieved_visual_feature_embedding, \
        retrieved_textual_feature_embedding, retrieved_label, label, item_id = zip(*batch)

    return (torch.tensor(visual_feature_embedding, dtype=torch.float))[:, :num_of_frames], torch.tensor(
        textual_feature_embedding,dtype=torch.float).unsqueeze(1), \
        (torch.tensor(similarity))[:, :num_of_retrieved_items], \
        (torch.tensor(retrieved_visual_feature_embedding, dtype=torch.float))[:, :num_of_retrieved_items, :num_of_frames, :], \
        (torch.tensor(retrieved_textual_feature_embedding, dtype=torch.float))[:, :num_of_retrieved_items, :], \
        (torch.tensor(retrieved_label, dtype=torch.float).unsqueeze(2))[:, :num_of_retrieved_items, :], torch.tensor(
        label, dtype=torch.float).unsqueeze(1)


class MyData(torch.utils.data.Dataset):

    def __init__(self, path):

        super().__init__()

        self.path = path

        self.dataframe = pd.read_pickle(path)

        self.visual_feature_embedding_list = self.dataframe['visual_feature_embedding_cls'].tolist()

        self.textual_feature_embedding_list = self.dataframe['textual_feature_embedding'].tolist()

        self.similarity_list = self.dataframe['retrieved_item_similarity_list'].tolist()

        self.retrieved_visual_feature_embedding_list = self.dataframe['retrieved_visual_feature_embedding_cls'].tolist()

        self.retrieved_textual_feature_embedding_list = self.dataframe['retrieved_textual_feature_embedding'].tolist()

        self.retrieved_label_list = self.dataframe['retrieved_label'].tolist()

        self.label_list = self.dataframe['label'].tolist()

        self.item_id_list = self.dataframe['item_id'].tolist()

    def __getitem__(self, index):

        visual_feature_embedding = self.visual_feature_embedding_list[index]

        textual_feature_embedding = self.textual_feature_embedding_list[index]

        similarity = self.similarity_list[index]

        retrieved_visual_feature_embedding = self.retrieved_visual_feature_embedding_list[index]

        retrieved_textual_feature_embedding = self.retrieved_textual_feature_embedding_list[index]

        retrieved_label = self.retrieved_label_list[index]

        label = self.label_list[index]

        item_id = self.item_id_list[index]

        return visual_feature_embedding, textual_feature_embedding, similarity, retrieved_visual_feature_embedding, \
            retrieved_textual_feature_embedding, retrieved_label, label, item_id

    def __len__(self):
        return len(self.dataframe)
