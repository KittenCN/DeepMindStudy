import os
from PIL import Image

ext = ['jpg','jpeg','png']
# files = os.listdir('.')
img_path = "/root/DeepMindStudy/emoji/data/test/"
def process_image(img_path, file, mwidth=512, mheight=512):
    filename = img_path + file
    image = Image.open(filename)
    w,h = image.size
    if w==mwidth and h==mheight:
        print(filename,'is OK.')
        return 
    if (1.0*w/mwidth) > (1.0*h/mheight):
        scale = 1.0*w/mwidth
        new_im = image.resize((int(w/scale), int(h/scale)), Image.ANTIALIAS)
    else:
        scale = 1.0*h/mheight
        new_im = image.resize((int(w/scale),int(h/scale)), Image.ANTIALIAS)     
    new_im.save(img_path + "new_" + file)
    new_im.close()
 
if __name__ == "__main__":
    for path, dirs, files in os.walk(img_path, topdown=False):
        file_list = list(files)
    for file in file_list:
        image_path = img_path + file
        if image_path.split('.')[-1] in ext:
            process_image(img_path, file, 512, 512)