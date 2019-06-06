import xml.etree.ElementTree as ET
import numpy as np
import cv2
import re
import os

TRAIN_SIZE = .8

def getPoints(nodes):
    """Convert from a list of xml Elements to a NumPy matrix of points
    Parameters
    ----------
    modes : list of Elements
	    A list of the x and y points of a polygone
    Returns
    -------
    numpy.ndarray
	    A NumPy's ndarray of the points to draw.
    """
    points = [(int(r[2]),r[1],int(p))  for (r,p) in [(re.search('^(x|y)(\d+)$', e.tag), e.text) for e in nodes]]
    return np.column_stack((getVector(points,'x'), getVector(points,'y')))

def getVector(points, l):
    """Extract from a list of points a sorted vector of an specific axis
    Parameters
    ----------
    points : Tuple
	    List of tuples whit the order, axis and point values
    l      : String
            The axis to extract
    Returns
    -------
    numpy.ndarray
	    A NumPy's ndarray of the axis values.
    """
    v = list(filter(lambda f: f[1] == l, points))
    v.sort(key = lambda v: v[0])
    return [p for (_,_,p) in v]
                           
def addPolygonShape(img, points):
    """Add a polygone Shape to an image
    Parameters
    ----------
    img    : numpy.ndarray
	  A NumPy's ndarray where we are drawing the polygons.
    points : numpy.ndarray
          A NumPy's with the points that represent the polygon.
    Returns
    -------
    numpy.ndarray
        A NumPy's ndarray with the image with the polygon drawed.
    """
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts],(0,255,0))
    return img

def createStack(orig, labels):
    """Create the AtoB/BtoA image
    Parameters
    ----------
    origin : numpy.ndarray
	  A NumPy's ndarray from cv2.imread as an input.
    lables   : xml.etree.ElementTree.Element
          The root element of the xml document
    Returns
    -------
    numpy.ndarray
        A NumPy's ndarray with the AtoB stacked image
    """
    height, width, channels = orig.shape
    bimg = np.zeros((height,width,channels), np.uint8)
    polyons = labels.findall("./object//polygon")
    for nodes in polyons:
        bimg = addPolygonShape(bimg, getPoints(list(nodes)))
    return np.hstack([orig,bimg])
        
if __name__ == '__main__':
    labels_path = "../datasets/labels"
    train_path = "../datasets/train"
    val_path = "../datasets/val"
    (_, _, filenames) = next(os.walk(labels_path))
    names = [os.path.splitext(f)[0] for f in list(filter(lambda x: x.endswith('xml'), filenames))]
    imgs = [(n, createStack(cv2.imread(f"{labels_path}/{n}.jpeg"), ET.parse(f"{labels_path}/{n}.xml").getroot())) for n in names]
    # Save the stacked image for training
    for n,i in imgs[:round(len(imgs)*TRAIN_SIZE)]:
        cv2.imwrite(f"{train_path}/{n}.jpg", i)
    # Save the stacked image for testing
    for n,i in imgs[round(len(imgs)*TRAIN_SIZE):]:
        cv2.imwrite(f"{val_path}/{n}.jpg", i)
