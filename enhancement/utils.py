# -*- coding: utf-8 -*-
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
    	To test the denoising run:
    ``python utils.py -i "../imgs/sample_image.jpg" -p denoise -s 10``
    	To test the kernel sliding run:
    ``python utils.py -i "../imgs/sample_image.jpg" -p kernel``
    	To test the dehazing run:
    ``python utils.py -i "../imgs/sample_image.jpg" -p dehaze``
	"""
	parser = argparse.ArgumentParser()

	processes = """Must be one of the following: [gamma][kernel][denoise][dehaze]"""

	parser.add_argument("-p", "--process", required=True,
						help=f"Process to be run. {processes}")
	parser.add_argument("-i", "--image", required=True,
						help="Path to input image.")
	parser.add_argument("-g", "--gamma", required=False, type=float,
						help="Gamma value to modify on input image.")
	parser.add_argument("-k", "--kernel", required=False,
						 help="Kernel to be slided over an image")
	parser.add_argument("-s", "--strength", required=False, type=float,
						 help ="Strength of denosing filter")

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

def slide_kernel(img, kernel=[[1,1,1],[1,20,1],[1,1,1]]):
	"""Slides a kernel over an input image
	Parameters
	----------
	img : numpy.ndarray
		A NumPy's ndarray from cv2.imread as an input.
	kernel : 2 dimensional table of floats (optional)
		Contains the kernel to be slided.
		By default it is set to [[1,1,1],[1,20,1],[1,1,1]] which provides sharpening
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray of an image with gamma modified.
	"""
	kernel_sum = 0
	for line in kernel:
		kernel_sum += sum(line)
	kernel = np.array(kernel).astype("float32") /kernel_sum
	return cv2.filter2D(img, -1, kernel)

def denoise(image, strength=10):
	"""Non Local Means Denosing based on OpenCV built in method
	Parameters
	----------
	img : numpy.ndarray
		A NumPy's ndarray from cv2.imread as an input.
	strength : integer (optional)
		defines strength of denoising operation 
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray of an image with gamma modified.
	"""
	return cv2.fastNlMeansDenoisingColored(img,None,strength,strength,7,21)

def get_dark_channel(img):
	"""subfunction for dehaze function (not to be used out of dehaze)
	provides dark channel of an RGB image (1 layer image composed of darkests of RGB pixels)
	Parameters
	----------
	img : numpy.ndarray
		A NumPy's ndarray from cv2.imread as an input.
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray of an image containing the dark channel.
	"""
	blue, green, red = cv2.split(img)
	dark = cv2.min(cv2.min(blue, green), red)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
	return cv2.erode(dark,kernel) #erode filter will remove small white elements from the dark channel (probably noise)
	#return dark

def get_atmospheric_light(img, dark):
	"""subfunction for dehaze function (not to be used out of dehaze)
	provides estimation of the atmosferic light (A) for an image
	Parameters
	----------
	img : numpy.ndarray
		A NumPy's ndarray from cv2.imread as an input.
	dark : numpy.ndarray
		A NumPy's ndarray containing the dark channel of img image (output of get_dark_channel function).
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray [1,3] containg BGR values for the estimated atmospheric light.
	"""
	img_size = img.shape[0]*img.shape[1]
	dark = dark.reshape(img_size,1)
	sorted_indexes = np.argsort(dark,0)
	sorted_indexes = sorted_indexes[int(0.9*img_size)::] #10% of brightest pixels in the dark channel
	blue, green, red = cv2.split(img)
	gray = (0.299*red + 0.587*green + 0.114*blue) 		#convert RGB to grey 
	gray = gray.reshape(img_size, 1)
	position = np.where(gray == max(gray[sorted_indexes]))		#localize brightest grey pixels over lightest dark channel spots
	position = position[0][0]	#position of the lightest grey pixel	
	img_vec = img.reshape(img_size,3)
		
	return (np.array(img_vec[position])).reshape(1,3)			#return RGB values corresponding to the brightest grey pixel - it is assumed it is the atmospheric light

def get_transmission(img, A):
	"""subfunction for dehaze function (not to be used out of dehaze)
	provides map of estimated transmission for an image
	Parameters
	----------
	img : numpy.ndarray
		A NumPy's ndarray from cv2.imread as an input.
	A : numpy.ndarray
		A NumPy's ndarray [1,3] containg BGR values for the atmospheric light.
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray containg map of estimated transmission.
	"""
	img_t = np.empty(img.shape, img.dtype)
	for layer in range(3):
		#if A[0,layer]==0:
		#	A[0,layer]=1
		img_t[:,:,layer] = img[:,:,layer]/A[0,layer]
	
	
	return 1-0.95*get_dark_channel(img_t)
   	
def refine_transmission(img,t_est):
	"""subfunction for dehaze function (not to be used out of dehaze)
	refines map of estimated transmission with soft matting method
	Parameters
	----------
	img : numpy.ndarray
		A NumPy's ndarray from cv2.imread as an input.
	t_est : numpy.ndarray
		A NumPy's ndarray with map of estimated transmission (output of get_transmission function).
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray containg refined map of transmission.
	"""
	
	r = 50
	eps = 0.0001
	
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = np.float64(gray)/255
	mean_I = cv2.boxFilter(gray,cv2.CV_64F,(r,r))
	mean_p = cv2.boxFilter(t_est, cv2.CV_64F,(r,r))
	mean_Ip = cv2.boxFilter(gray*t_est,cv2.CV_64F,(r,r))
	cov_Ip = mean_Ip - mean_I*mean_p
	mean_II = cv2.boxFilter(gray*gray,cv2.CV_64F,(r,r))
	var_I   = mean_II - mean_I*mean_I
	a = cov_Ip/(var_I + eps)
	b = mean_p - a*mean_I
	mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
	mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))
	
	return mean_a*gray + mean_b

def recover(img, t, A):
	"""subfunction for dehaze function (not to be used out of dehaze)
	recovers dehazed image for a hazed one
	Parameters
	----------
	img : numpy.ndarray
		A NumPy's ndarray from cv2.imread as an input.
	t : numpy.ndarray
		A NumPy's ndarray with map of transmission (output refine_transmission function).
	A : numpy.ndarray
		A NumPy's ndarray [1,3] containg BGR values for the atmospheric light.
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray containg dehazed BGR image.
	"""
	img_t = np.empty(img.shape, img.dtype)
	t = cv2.max(t,0.1) 	#make sure that transmission is not 0
	
	for layer in range(3):
		img_t[:,:,layer] = (img[:,:,layer]-A[0,layer])/t + A[0,layer]
	
	img_t[img_t<0]=0
	
	return img_t
	
def dehaze(img):
	"""	dehazing of an image based on Dark Channel Prior method
	Parameters
	----------
	img : numpy.ndarray
		A NumPy's ndarray from cv2.imread as an input.
	
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray with the dahazed image.
	"""		
	img_norm = img.astype('float64')/255
	dark_channel = get_dark_channel(img_norm)
	A = get_atmospheric_light(img_norm, dark_channel)
	t_est = get_transmission(img_norm, A)
	t = refine_transmission(img, t_est)
	modified_64 = recover(img_norm, t, A)
	modified = (modified_64*255).astype('uint8')
	
	return modified

if __name__ == '__main__':
	args = parser()
	
	try:
		# Test modify_gamma:
		if args['process'] == 'gamma':
			img = cv2.imread(args['image'])
			gamma = args['gamma'] or 1.0
			modified = modify_gamma(img, gamma=gamma)
			cv2.imshow(f"Gamma = {gamma}", np.hstack([img, modified]))
			cv2.imwrite("modified.jpg", modified)
			cv2.waitKey(0)
		
		# Test denoising:
		elif args['process'] == 'denoise':
			img = cv2.imread(args['image'])
			strength = args['strength'] or 10.0
			modified = denoise(img, strength=strength)
			cv2.imshow(f"Denoised with strength = {strength}", np.hstack([img, modified]))
			cv2.imwrite("modified.jpg", modified)
			cv2.waitKey(0) 
		
		# Test slide_kernel:
		elif args['process'] == 'kernel':
			img = cv2.imread(args['image'])
			kernel = args['kernel'] or [[1,1,1],[1,20,1],[1,1,1]]
			modified = slide_kernel(img, kernel=kernel)
			cv2.imshow(f"Treated with kernel = {kernel}", np.hstack([img, modified]))
			cv2.imwrite("modified.jpg", modified)
			cv2.waitKey(0)
		
		# Test dehaze:
		elif args['process'] == 'dehaze':
			img = cv2.imread(args['image'])
			modified = dehaze(img)
			cv2.imshow("Dehazed", np.hstack([img, modified]))
			cv2.imwrite("modified.jpg", modified)
			cv2.waitKey(0)
			
		# Error setting testing process:
		else:
			msg = "Incorrect process. Please verify valid processes with -h flag."
			raise ValueError(msg)

	except ValueError as error:
		print(error)
