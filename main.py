from mangum import Mangum
import uvicorn
from fastapi import FastAPI
import torch
import clip
from PIL import Image
import json
import requests
from io import BytesIO


app = FastAPI()

MODEL_NAME = "ViT-B/32"
DEVICE = "cpu"
model_cache = {}
FEATURES = json.load(open("features.json", "r"))


def _init_model():
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE, download_root="/tmp")
    model_cache['model'] = model
    model_cache['preprocess'] = preprocess

def get_k_closest_images(input_feature, features: dict, k=5):
    distances = {}
    for image_name, feature in features.items():
        distance = torch.linalg.norm(input_feature - torch.tensor(feature))
        distances[image_name] = distance.item()

    distances = sorted(distances.items(), key=lambda x: x[1])
    return distances[:k]


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/recommendation/image")
def get_recommendation_from_image(url: str, k: int = 5):
    if 'model' not in model_cache:
        _init_model()
    response = requests.get(url)
    web_image = Image.open(BytesIO(response.content))

    model = model_cache['model']
    preprocess = model_cache['preprocess']

    image = preprocess(web_image).unsqueeze(0).to("cpu")

    with torch.no_grad():
        image_features = model.encode_image(image)

    return get_k_closest_images(image_features, FEATURES, k)


@app.get("/recommendation/text")
def get_recommendation_from_text(text: str, k: int = 5):
    if 'model' not in model_cache:
        _init_model()
    model = model_cache['model']
    preprocess = model_cache['preprocess']

    text = clip.tokenize(text).to(DEVICE)

    with torch.no_grad():
        feature = model.encode_text(text)

    return get_k_closest_images(feature, FEATURES, k)


handler = Mangum(app, lifespan="off")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
