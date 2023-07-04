import os

from PIL import Image

def save_images(images, imgdir, startid=1, nwidth=5):
    os.makedirs(imgdir, exist_ok=True)
    for img in images:
        img = Image.fromarray(img*255)
        img.convert('L').save(os.path.join(imgdir, str(startid).zfill(nwidth) + ".png"))
        startid += 1
