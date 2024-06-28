from PIL import Image

def up(img_path, img_name):
    original_image = Image.open(img_path+img_name)
    original_image.save(f"{img_path}up_{img_name}", "JPEG", quality=100)
