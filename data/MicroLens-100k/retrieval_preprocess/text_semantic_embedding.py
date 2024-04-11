

import pandas as pd
from angle_emb import AnglE, Prompts
from tqdm import tqdm


def loading_model():

    local_model_path = r"UAE-Large-V1"

    angle = AnglE.from_pretrained(local_model_path, pooling_strategy='cls').cuda()

    angle.set_prompt(prompt=Prompts.C)

    return angle


def convert_text_to_embedding(angle, text):

    vec = angle.encode({'text': text}, to_numpy=True)

    return vec

