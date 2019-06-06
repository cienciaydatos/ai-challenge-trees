import os
import sys
import argparse
import numpy as np
import xml.etree.ElementTree as et
from itertools import zip_longest
from PIL import Image, ImageDraw
from tqdm import tqdm

def grouper(iterable, n=2):
    args = [iter(iterable)] * n
    return zip_longest(*args)

def parse_polygon(etelem):
    points = []
    for elem in grouper(etelem, 2):
        points.append(tuple(map(lambda x: int(x.text), elem)))
    return points

def parse_folder(folder):
    w, h = 1000, 1000
    for file in tqdm(os.listdir(folder)):
        if 'xml' in file:
            tree = et.parse(folder + file)
            root = tree.getroot()

            filename = root.findall('filename')[0].text
            polygon = []
            name_list = []

            for lmobj in root.findall('object'):
                name = lmobj.find('name').text
                name_list.append(name)
                pol = parse_polygon(lmobj.find('polygon'))
                polygon.append(pol)

            img = Image.new('RGB', [w, h], (0,0,0))
            for poly, name in zip(polygon,name_list):
                color = (255,0,0) if 'not' in name.lower() else (0,255,0)
                if (len(poly) < 3): continue
                ImageDraw.Draw(img).polygon(poly, outline=color, fill=color)
                mask = np.array(img)

            filename = filename.replace('.jpeg','_m.jpeg')
            img.save(folder + filename, 'JPEG')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', help='target folder')
    args = parser.parse_args()
    parse_folder(args.target)