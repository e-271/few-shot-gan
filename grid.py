from PIL import Image
import argparse
import os
import math
from glob import glob

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('path', type=str, help='Path to image dir.')
parser.add_argument('res', type=int, help='Resolution.')
parser.add_argument('nimg', type=int, help='First n img.')

args = parser.parse_args()
path, res, nimg = args.path, args.res, args.nimg

imgs = glob(path + '*.jpg')[:nimg]
n=len(imgs)
cols = int(math.ceil(math.sqrt(n)))
rows = int(math.ceil(n / cols))

grid = Image.new('RGB', (res * cols, res * rows), (255, 255, 255))

r, c = 0, 0

for img_path in imgs:
    print(img_path)
    im = Image.open(img_path)
    im = im.resize((res, res))
    grid.paste(im, (c, r))
    c += res
    if c + res > res * cols:
        r += res
        c = 0
    print(r, c)


name = path.split('/')[-1]
grid.save('%s_grid.png' % name)
         

