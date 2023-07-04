import os

def save_images(images, imgdir, startid=1, nwidth=5):
    from PIL import Image
    os.makedirs(imgdir, exist_ok=True)
    for img in images:
        img = Image.fromarray(img*255)
        img.convert('L').save(os.path.join(imgdir, str(startid).zfill(nwidth) + ".png"))
        startid += 1
