import json
import skimage
import matplotlib.pyplot as plt
import argparse
from model import UNET

argparser = argparse.ArgumentParser(
    description='end-to-end U-net training')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to config file')

with open(argparser.parse_args().conf) as raw:
    config = json.load(raw)

unet = UNET(config)
unet.train()
