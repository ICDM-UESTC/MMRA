import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def loading_model():

    local_path = r"BLIP"

    processor = BlipProcessor.from_pretrained(local_path)

    model = BlipForConditionalGeneration.from_pretrained(local_path).to("cuda")

    return processor, model


def convert_image_to_text(processor, model, image_path):

    raw_image = Image.open(image_path).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)

    text = processor.decode(out[0], skip_special_tokens=True)

    return text


def process_row(item_id, path, processor, model):

    current_text_list = []

    for j in range(0, 10):

        image_path = os.path.join(path, f"{item_id}_{j}.jpg")

        text = convert_image_to_text(processor, model, image_path)

        current_text_list.append(text)

    return current_text_list
