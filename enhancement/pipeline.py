# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:54:03 2019

@author: Cristian Vargas and Lukasz Kaczmarek
"""

import cv2
import os
from utils import dehaze, gamma, unsharp, denoise, kernel, clahe


class PipeItem(object):
    
    def __init__(self, item, **kwargs):
        self.item = item
        self.kwargs = kwargs
        
    def run(self, image):
        return self.item(image, **self.kwargs)

class Pipeline(object):
	
	def __init__(self, pipeline=[]):
		
		self.images = []
		self.pipeline = pipeline
		self.outputPath = ""
		self.inputPath = ""
		self.outputFIleType = "jpg"
	
	def setOutputFileType(self, fileType="jpg"):
		
		if fileType in ("jpg","png", "tif"):
			self.outputFIleType = fileType
		else:
			print("File type must be jpg, png or tif")
	
	def addImage(self, imagePath):
		self.images.append(imagePath)

	def addFolder(self, path):
		self.inputPath = path
		"""
		here add method to add images to self.images with os.path
		"""
	
	def addPipeItem(self, item):
		self.pipeline.append(item)
	
	
	def setPipeline(self, pipeline):
		self.pipeline = pipeline
	
	def saveModified(self, image):
		name = "modified." + format
		try:
			cv2.imwrite( name, image )
		except IOError as error:
			print(error)
			
	def run(self, save_files = True, display_steps = False):
		
		for image in self.images:
			print("processing image: " + image)
			try:
				temp = cv2.imread(image)
			except IOError as error:
				print(error)
				
			mod_step = 0
			for item in self.pipeline:
				if display_steps:
					cv2.imshow("modification"+str(mod_step)+" "+image, temp)
				temp = item.run(temp)
			
			if display_steps:
				cv2.imshow("final modification: "+image, temp)
					
			if save_files:
				self.saveModified(temp)	

pipeline = [
    PipeItem(dehaze),
    PipeItem(gamma, gamma=1.8),
    PipeItem(kernel, kernel=[[-1,-1,-1],[-1,20,-1],[-1,-1,-1]]),
    PipeItem(clahe),
]
	
imagePath = r"C:/Users/lukasz.kaczmarek/source/Python/test.tif"

pipeline = Pipeline()
pipeline.addImage(imagePath)
pipeline.setPipeline(pipeline)	
pipeline.run(save_files = True, display_steps = False)
