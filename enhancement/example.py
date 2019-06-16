# -*- coding: utf-8 -*-

"""

this file shows how to use image enhancement librares developed by task 3 in your code

@author: Lukasz Kaczmarek

"""

from  pipeline import PipeItem, Pipeline
from utils import dehaze, clahe, gamma, kernel  #here import filters which will be used (available: kernel, dehaze, denoise, unsharp, gamma, clahe)

filter_sequence = [											#sequence of filters to be appiled to an image
			PipeItem(dehaze),
			PipeItem(gamma, gamma=1.8),
			PipeItem(kernel, kernel=[[-1,-1,-1],[-1,20,-1],[-1,-1,-1]]),
			PipeItem(clahe)
			]

pipeline = Pipeline(pipeline = filter_sequence)				#create an instance of pipeline.Pipeline
filtered_image = pipeline.process(image = r"c:\users\lukasz.kaczmarek\source\python\Australia\test.tif", display_steps=True)
