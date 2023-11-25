import os
import torch
import clip
from PIL import Image
import json
from tqdm import tqdm

DEVICE = "cpu"
MODEL_NAME = "ViT-B/32"
def extract_features():
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)

    # extract features from images and save them to a json file
    features = {}
    images_dir = os.path.join(os.getcwd(), "images")

    for image_name in tqdm(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_name)

        with torch.no_grad():
            image_features = model.encode_image(image)
            features[image_name] = image_features.cpu().numpy().tolist()

    with open("features.json", "w") as fp:
        json.dump(features, fp)

def chech_feature_counts():
    with open("features.json", "r") as fp:
        features = json.load(fp)

    print(len(features))


if __name__ == "__main__":
    chech_feature_counts()
