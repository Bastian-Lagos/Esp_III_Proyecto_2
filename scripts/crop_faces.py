from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os, glob
from tqdm import tqdm

mtcnn = MTCNN(image_size=160, margin=14)

def crop_folder(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    paths = glob.glob(os.path.join(src_folder, "*"))
    for p in tqdm(paths):
        try:
            out_path = os.path.join(dst_folder, os.path.basename(p))
            img = Image.open(p).convert('RGB')
            mtcnn(img, out_path)
        except Exception as e:
            print("ERROR", p, e)

if __name__ == "__main__":
    crop_folder("data/me", "data/cropped/me")
    crop_folder("data/not_me", "data/cropped/not_me")