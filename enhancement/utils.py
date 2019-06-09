# -*- coding: utf-8 -*-
import argparse
import inspect
import numpy as np
import cv2
from scipy.ndimage.filters import median_filter
 
SAMPLE_SIZE=100
PROCESS=["gamma","kernel","denoise","dehaze", "unsharp", "clahe"]

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

	processes = f"Must be one of the following: {', '.join(PROCESS)}"

	parser.add_argument("-a", "--all", required=False, default=False, action='store_true',
						 help ="Compare all")
	parser.add_argument("-e", "--sequence", required=False, default=False, action='store_true',
						 help ="Applies sequence of filters")
	parser.add_argument("-c", "--clip_limit", required=False, type=float, default=2.0,
						 help ="Clahe clip Limit")
	parser.add_argument("-i", "--image", required=True,
						help="Path to input image.")
	parser.add_argument("-g", "--gamma", required=False, type=float, default=1.0,
						help="Gamma value to modify on input image.")
	parser.add_argument("-k", "--kernel", required=False, default='1 1 1;1 20 1;1 1 1',
						 help="Kernel to be slided over an image")
	parser.add_argument("-p", "--process", required=False,
						help=f"Process to be run. {processes}")
	parser.add_argument("-q", "--sigma", required=False, type=int, default=8,
						 help ="Unsharp Sigma")
	parser.add_argument("-s", "--strength", required=False, type=float, default=10,
						 help ="Strength of denosing filter")
	parser.add_argument("-t", "--tile_grid_size", required=False, type=int, default=8,
						 help ="Clahe Tile Grid Size")
	parser.add_argument("-x", "--ustrength", required=False, type=float, default=2,
						 help ="Unsharp Strength")
	parser.add_argument("-y", "--apply", required=False, type=int, default=1,
						 help ="Clahe Apply")
	parser.add_argument("-z", "--zoom_from", required=False, type=int,
						 help ="Point to zoom from")
						
	args = vars(parser.parse_args())

	if not args['all'] and args['process'] is None:
		parser.error('without -a, -p flag is required')
	return args

def gamma(image, gamma=1.0):
	"""Adjusts gamma value on input image.
	Parameters
	----------
	image : numpy.ndarray
		A NumPy's ndarray from cv2.imread as an input.
	gamma : float (optional)
		A float containing the gamma value to be adjusted.
		By default it is set to 1.0.
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray of an image with gamma modified.
	"""
	gamma = gamma or 1.0
	power = (1.0 / gamma)
	table = [((i / 255.0) ** power) * 255.0 for i in np.arange(0, 256)]
	table = np.array([table]).astype("uint8")
 
	return cv2.LUT(image, table)

def kernel(image, kernel=[[1,1,1],[1,20,1],[1,1,1]]):
	"""Slides a kernel over an input image
	Parameters
	----------
	image : numpy.ndarray
		A NumPy's ndarray from cv2.imread as an input.
	kernel : 2 dimensional table of floats (optional)
		Contains the kernel to be slided.
		By default it is set to [[1,1,1],[1,20,1],[1,1,1]] which provides sharpening
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray of an image with gamma modified.
	"""
	kernel = np.matrix(kernel).tolist() if type(kernel) is str else kernel or [[1,1,1],[1,20,1],[1,1,1]]
	kernel_sum = 0
	for line in kernel:
		kernel_sum += sum(line)
	kernel = np.array(kernel).astype("float32") /kernel_sum
	return cv2.filter2D(image, -1, kernel)

def denoise(image, strength=10):
	"""Non Local Means Denosing based on OpenCV built in method
	Parameters
	----------
	image : numpy.ndarray
		A NumPy's ndarray from cv2.imread as an input.
	strength : integer (optional)
		defines strength of denoising operation 
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray of an image with gamma modified.
	"""
	strength = strength or 10
	return cv2.fastNlMeansDenoisingColored(image,None,strength,strength,7,21)

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
	
def dehaze(image):
	"""	dehazing of an image based on Dark Channel Prior method
	Parameters
	----------
	image : numpy.ndarray
		A NumPy's ndarray from cv2.imread as an input.
	
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray with the dahazed image.
	"""		
	img_norm = image.astype('float64')/255
	dark_channel = get_dark_channel(img_norm)
	A = get_atmospheric_light(img_norm, dark_channel)
	t_est = get_transmission(img_norm, A)
	t = refine_transmission(image, t_est)
	modified_64 = recover(img_norm, t, A)
	modified = (modified_64*255).astype('uint8')
	
	return modified

def unsharp(image, sigma=8, ustrength=2):
	"""Unsharp masking on input image.
	Parameters
	----------
	image : numpy.ndarray
		A NumPy's ndarray from cv2.imread as an input.
	sigma : integer (optional)
		A integer containing the size of the footprint to apply to a median filter to the image.
		By default it is set to 5.
	ustrength : float (optional)
		A float containing the amount of the Laplacian version of the image to add or take
		By default it is set to add 0.8.
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray of an image with gamma modified.
	"""
	sigma = sigma or 8
	ustrength = ustrength or 2
	sharp = np.zeros_like(img)
	for i in range(3):
		sharp[:,:,i] = unsharp_channel(img[:,:,i], sigma, ustrength)
    
	return sharp
	
def unsharp_channel(chan, sigma, strength):
	"""Unsharp masking on input image channel.
	Parameters
	----------
	chan : numpy.ndarray
		A NumPy's ndarray from the channel to sharp.
	sigma : integer 
		A integer containing the size of the footprint to apply to a median filter to the image.
	ustrength : float 
		A float containing the amount of the Laplacian version of the image to add or take
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray of a channel with gamma modified.
	"""
	# Median filtering
	image_mf = median_filter(chan, sigma)

    # Calculate the Laplacian
	lap = cv2.Laplacian(image_mf,cv2.CV_64F)

    # Calculate the sharpened image
	sharp = chan-strength*lap

    # Saturate the pixels in either direction
	sharp[sharp>255] = 255
	sharp[sharp<0] = 0
	
	return sharp

def clahe(image, clip_limit=2.0, tile_grid_size=8, apply=1):
    """Contrast Limited Adaptive Histogram Equalization
    Based on https://stackoverflow.com/questions/25008458/how-to-apply-clahe-on-rgb-color-images
    Parameters
    ----------
    img : numpy.ndarray
        A NumPy's ndarray from cv2.imread as an input.
    clip_limit: float
        Clip threshold for CLAHE
    tile_grid_size: int
        Size of the grid to perform CLAHE calculation
    apply: int
        Set this if you want to reapply clahe to reduce the clip overshoot
    Returns
    -------
    numpy.ndarray
        A NumPy's ndarray of an image 
    """
    clip_limit = clip_limit or 2.0
    tile_grid_size = tile_grid_size or 8
    apply = apply or 1
    he = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
    )
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    for _ in range(apply):
        lab_planes[0] = he.apply(lab_planes[0])  # Lightness component
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
	
def compare(title, orig, mod, zoom_from=0):
	"""Compare portions of an image of size (SAMPLE_SIZE x SAMPLE_SIZE) starting from the zoom_from point.
	Parameters
	----------
	title : string
		The title of the window.
	orig : numpy.ndarray
		The original image.
	mod : numpy.ndarray
		The image enhanced.
	zoom_from : integer (optional)
		A integer containing starting point to crop and apply the zoom.
		By default it is set to 0.
	"""
	cv2.imwrite("modified.jpg", mod)
	zoom_from = zoom_from or 0
	h, w, c, nh, nw = get_shape_and_new_dim(orig)
	zoom = np.hstack([
		get_zoom(orig, zoom_from, nh, nw),
		get_zoom(mod, zoom_from, nh, nw)
		])
	draw_zoom_area(orig,zoom_from)
	draw_zoom_area(mod,zoom_from)
	comp = np.hstack([cv2.resize(orig, (nh,nw)), cv2.resize(mod, (nh,nw))])
	
	cv2.imshow(title, np.vstack([zoom,comp]))
	cv2.waitKey(0)
	
def compare_all(orig, args):
	"""Make comparations of all the methods in the array PROCESS.
	Parameters
	----------
	orig : numpy.ndarray
		The original image.
	args : Dictionary
		The parsed dictionary of the script arguments.
	"""
	h, w, c, nh, nw = get_shape_and_new_dim(orig)
	imgs = [get_zoom(globals()[p](*get_process_opts(args, p)), args['zoom_from'], nh, nw) for p in PROCESS]
	labels = PROCESS
	if (len(imgs) % 2) == 1: 
		imgs.append(np.zeros((nh,nw,c), np.uint8))
		labels.append("N/A")
	vs = [np.vstack([col[0],col[1]]) for col in  np.array(imgs).reshape(len(imgs)//2,2,nh,nw,c)]
	draw_zoom_area(orig,args['zoom_from'])
	vs.append(orig)
	cv2.imshow("All process: " + ', '.join(labels), np.hstack(vs))
	cv2.waitKey(0)

def get_shape_and_new_dim(img):
	"""Gets the shape of the image and the size of the zoom portion of the sample (50% of the orignal label)
	Parameters
	----------
	img : numpy.ndarray
		The original image.
	Returns
	-------
	tuple
		(height, width, channels, new_heigth, new_width)
	"""
	height, width, channels = img.shape
	new_heigth = height//2
	new_width = width//2
	return (height, width, channels, new_heigth, new_width)

def get_zoom(img, zoom_from, new_heigth, new_width):
	"""Build the zoom portion of the image
	Parameters
	----------
	img : numpy.ndarray
		The original image.
	zoom_from : integer 
		A integer containing starting point to crop and apply the zoom.
	new_heigth : integer 
		The new height of the image to generate.
	new_width : integer 
		The new width of the image to generate.
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray of the image generated.
	"""
	return cv2.resize(img[zoom_from:zoom_from+SAMPLE_SIZE,zoom_from:zoom_from+SAMPLE_SIZE], (new_heigth,new_width))

def get_process_opts(args, process=None):
	"""Build an array of the available options for the process
	Parameters
	----------
	args : Dictionary
		The arguments passed and parsed to the script
	process : string (optional)
		The process to build the array of arguments, if None, the functions use the value of the key 'process' of the args dictionary.
		By default it is set to None.
	Returns
	-------
	Array
		The array of parameters to pass to the function.
	"""
	process = process or args['process']
	return [img if a == 'image' else args[a] for a in inspect.getfullargspec(globals()[process]).args]
	
def get_window_title(args, process=None):
	"""Build the title to show in a image window
	Parameters
	----------
	args : Dictionary
		The arguments passed and parsed to the script
	process : string (optional)
		The process to build title for, if None, the functions use the value of the key 'process' of the args dictionary.
		By default it is set to None.
	Returns
	-------
	String
		The string of the title
	"""
	process = process or args['process']
	return process.title() + ': ' + ', '.join([f"{a} = {args[a]}" for a in inspect.getfullargspec(globals()[process]).args if a != 'image'])
	
def draw_zoom_area(img, zoom_from):
	"""Draw a rectangle of the zoom area in the source image
	Parameters
	----------
	img : numpy.ndarray
		The target image.
	zoom_from : integer 
		A integer containing starting point to apply the zoom rectangle.
	Returns
	-------
	numpy.ndarray
		A NumPy's ndarray of the image generated.
	"""
	cv2.rectangle(img,(zoom_from,zoom_from),(zoom_from+SAMPLE_SIZE,zoom_from+SAMPLE_SIZE),(255,0,0),3)

if __name__ == '__main__':
	args = parser()
	
	try:
		img = cv2.imread(args['image'])
		if args['all']:
			compare_all(img, args)
		elif args['process'] in locals():
			f_args = get_process_opts(args)
			mod = locals()[args['process']](*f_args)
			compare(get_window_title(args), img, mod, args['zoom_from'])
		else:
			msg = "Incorrect process. Please verify valid processes with -h flag."
			raise ValueError(msg)

	except ValueError as error:
		print(error)
