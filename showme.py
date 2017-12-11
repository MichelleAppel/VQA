import os
from PIL import Image
from urllib import request

def show_image(imgid2info, id):
    img_name = 'temp-image.jpg'
    request.urlretrieve(imgid2info[str(id)]['flickr_url'], img_name)

    img = Image.open(img_name)
    img.show()

    os.remove(img_name)
    img.close()