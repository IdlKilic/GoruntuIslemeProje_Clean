import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json


def load_model_config():

    model_config_path = os.path.join("./model", "model_config.json")
    char_map_path = os.path.join("./model", "char_map.json")

    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model konfigürasyon dosyası bulunamadı: {model_config_path}")

    if not os.path.exists(char_map_path):
        raise FileNotFoundError(f"Karakter haritası dosyası bulunamadı: {char_map_path}")

    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    with open(char_map_path, "r") as f:
        char_map = json.load(f)

    return model_config, char_map


def predict_image(model_path, image_path):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path}")

    model_config, char_map = load_model_config()
    max_length = model_config["max_length"]
    vocab_size = model_config["vocab_size"]

    model = load_model(model_path)

    image = Image.open(image_path).convert("L").resize((256, 64))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0) 
    image_array = np.expand_dims(image_array, axis=-1)  

    prediction = model.predict(image_array)
    predicted_indices = np.argmax(prediction, axis=-1)[0]

    index_to_char = {int(k): v for k, v in char_map["index_to_char"].items()}
    predicted_text = "".join([index_to_char.get(idx, "") for idx in predicted_indices if idx > 0])

    if not predicted_text:
        print("Uyarı: Model tahmin yapamadı veya tahmin edilen metin boş.")
        return ""

    return predicted_text


