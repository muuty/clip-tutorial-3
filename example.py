import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("images/0.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["스트릿"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    print("image features : ", image_features.shape)
    print("text features : ", text_features)
