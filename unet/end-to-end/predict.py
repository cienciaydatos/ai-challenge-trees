import os
import json
import skimage
import argparse
import matplotlib.pyplot as plt
from model import UNET

argparser = argparse.ArgumentParser(
    description='U-net predictions')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to config file')

argparser.add_argument(
    '-i',
    '--input',
    help='path to input image')

argparser.add_argument(
    '-w',
    '--weight',
    help='path to input weight')


with open(argparser.parse_args().conf) as raw:
    config = json.load(raw)

img_path = argparser.parse_args().input
wei_path = argparser.parse_args().weight

unet = UNET(config)
unet.load_weights(wei_path)

I = skimage.io.imread(img_path)
I = skimage.transform.resize(I, (388, 388))
Ip = skimage.util.pad(I, ((92, 92), (92, 92), (0, 0)), 'constant')
out = unet.model.predict(Ip.reshape(1, *Ip.shape)).argmax(axis=-1).squeeze()

plt.figure(figsize=(10, 10))
plt.imshow(I)
plt.imshow(1-out, alpha=0.2)
plt.axis('off')
plt.show()
