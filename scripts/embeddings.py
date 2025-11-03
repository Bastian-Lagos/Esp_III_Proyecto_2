import os, glob
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import csv

model = InceptionResnetV1(pretrained='vggface2', classify=False).eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def embed_image(img_path):
    img = Image.open(img_path).convert('RGB')
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(x)
    return emb[0].cpu().numpy()

def build_embeddings(cropped_root="data/cropped", out_npz="data/embeddings.npz", csv_out="data/embeddings.csv"):
    X = []
    y = []
    meta = []
    for label_name, label in [("me",1), ("not_me",0)]:
        folder = os.path.join(cropped_root, label_name)
        for p in tqdm(glob.glob(os.path.join(folder, "*"))):
            try:
                e = embed_image(p)
                X.append(e)
                y.append(label)
                meta.append(p)
            except Exception as ex:
                print("err", p, ex)
    X = np.stack(X)
    y = np.array(y)
    np.savez(out_npz, X=X, y=y, meta=np.array(meta))
    with open(csv_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["path","label"])
        for m,lab in zip(meta,y):
            w.writerow([m,lab])
    print("Saved", out_npz, csv_out)

if __name__ == "__main__":
    build_embeddings()
