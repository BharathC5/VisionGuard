import cv2
import numpy as np
import os

IMG_SIZE = 300

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        if mask.any():
            return img[np.ix_(mask.any(1), mask.any(0))]
        else:
            return img
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > tol
        if mask.any():
            img1 = img[np.ix_(mask.any(1), mask.any(0))]
            return img1
        else:
            return img

def preprocess_aptos(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1) crop away dark borders
    img = crop_image_from_gray(img)

    # 2) resize to square
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 3) Ben Graham style local normalization
    #    (sharp foreground, subtract blurred background)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=10)
    img = cv2.addWeighted(img, 4, blur, -4, 128)

    # 4) optional CLAHE on L-channel
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img

def preprocess_folder(src_root, dst_root):
    os.makedirs(dst_root, exist_ok=True)
    for cls in os.listdir(src_root):
        src_dir = os.path.join(src_root, cls)
        dst_dir = os.path.join(dst_root, cls)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in os.listdir(src_dir):
            if not fname.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                continue
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            img = preprocess_aptos(src_path)
            cv2.imwrite(dst_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# Example:
# preprocess_folder("data/train", "data_pp/train")
# preprocess_folder("data/test",  "data_pp/test")
