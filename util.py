import os

def restore_block(block, param_file, ctx):
    if os.path.isfile(param_file):
        block.load_params(param_file, ctx)
        return True
    return False

def save_images(images, imgdir, startid=1, nwidth=5):
    from PIL import Image
    os.makedirs(imgdir, exist_ok=True)
    for img in images:
        img = Image.fromarray(img*255)
        img.convert('L').save(os.path.join(imgdir, str(startid).zfill(nwidth) + ".png"))
        startid += 1
    print(len(images), "test images written to", imgdir)
