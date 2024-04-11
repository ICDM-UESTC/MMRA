import pandas as pd
import torch
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import os


def load_vit_model():

    processor = ViTImageProcessor.from_pretrained(
        r'vit-base-patch16-224-in21k')

    model = ViTModel.from_pretrained(
        r'vit-base-patch16-224-in21k')

    return processor, model


def vit_visual_feature_extraction(processor, model, image_path):

    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)

    cls_output = outputs.last_hidden_state[:, 0, :]

    return (cls_output[0]).tolist()


