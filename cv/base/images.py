"""
The MIT License (MIT)

Copyright (c) 2015 Daniel von Poschinger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import glob
import matplotlib.image as pltimg
import matplotlib.pyplot as plt


def get_paths_of_images(
	path, 
	imgtype, 
	verbose=True
	):
	images_path = str(path) + '/*.' + imgtype
	paths = glob.glob(images_path)
	if verbose: print "images: found", len(paths), "images of type (", imgtype, ") in", path
	return paths



def read_image(
	image_path, 
	verbose=False
	):
	img = pltimg.imread(image_path).astype(float)
	img /= img.max()
	if verbose: print "images: read", image_path
	plt.close()
	return img


