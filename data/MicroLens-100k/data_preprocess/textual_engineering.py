import pandas as pd
from angle_emb import AnglE
from tqdm import tqdm
from torchinfo import summary


def load_angle_bert_model():

    angle = AnglE.from_pretrained(
        r'angle-bert-base-uncased-nli-en-v1',
        pooling_strategy='cls_avg').cuda()

    return angle


def angle_bert_textual_feature_extraction(angle, text):

    text_embedding = angle.encode(text, to_numpy=True)

    return (text_embedding[0]).tolist()
