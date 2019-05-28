import argparse

import numpy as np
import cv2
 

def parser():
	"""Creates a parser for process to be run.

	Returns
	-------
	args : dictionary
		A dictionary containing command line arguments to test processes.
	
	Examples
    --------
	To test the gamma modifier run:
    ``python utils.py -i "../imgs/sample_image.jpg" -p gamma -g 1.8``

	"""
	parser = argparse.ArgumentParser()

	processes = """Must be one of the following: [gamma]"""

	parser.add_argument("-p", "--process", required=True,
						help=f"Process to be run. {processes}")
	parser.add_argument("-i", "--image", required=True,
						help="Path to input image.")
	parser.add_argument("-g", "--gamma", required=False, type=float,
						help="Gamma value to modify on input image.")

	args = vars(parser.parse_args())

	return args


def modify_gamma(img, gamma=1.0):
	"""Adjusts gamma value on input image.

	Parameters
	----------
	img : numpy.ndarray
		A NumPy's ndarray from cv2.imread as an input.
	gamma : float (optional)
		A float containing the gamma value to be adjusted.
		By default it is set to 1.0.

	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray of an image with gamma modified.
	"""
	power = (1.0 / gamma)
	table = [((i / 255.0) ** power) * 255.0 for i in np.arange(0, 256)]
	table = np.array([table]).astype("uint8")
 
	return cv2.LUT(img, table)


if __name__ == '__main__':
	args = parser()
	
	try:
		# Test modify_gamma:
		if args['process'] == 'gamma':
			img = cv2.imread(args['image'])
			gamma = args['gamma'] or 1.0
			modified = modify_gamma(img, gamma=gamma)
			cv2.imshow(f"Gamma = {gamma}", np.hstack([img, modified]))
			cv2.waitKey(0)
		
		# Error setting testing process:
		else:
			msg = "Incorrect process. Please verify valid processes with -h flag."
			raise ValueError(msg)

	except ValueError as error:
		print(error)