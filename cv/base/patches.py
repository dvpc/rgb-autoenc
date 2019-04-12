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

import numpy as N


def prepare_patch(
	patch,
	mode,
	subtract_mean=True,
	chance_to_sawp_axes=50,
	normalize=True,
	):
	if mode=='rgb':
		pass
	elif mode=='rg':
		patch[:,:,0:2]
	elif mode=='rg_vs_b':
		N.array([(patch.T[0]+patch.T[1])/2., patch.T[2]]).T
	elif mode=='lum':
		patch = (patch.T[0]+patch.T[1]+patch.T[2])/3.
	else:
		raise 'mode not any of {rgb,rg,rg_vs_b,lum}!'
	if subtract_mean:
		patch -= N.mean(patch, dtype=patch.dtype)
	else:
		patch /= 1.5#2.
	if N.random.randint(0, 100) > 100-chance_to_sawp_axes:
		patch = N.swapaxes(patch, 0, 1)
	if mode=='lum':
		patch = patch.reshape(patch.shape[0]*patch.shape[1])
		return N.copy(patch)
	else:
		'''remove alpha channel'''
		if patch.shape[2] > 3:
			patch = patch.T[0:3].T
		patch = patch.reshape(patch.shape[0]*patch.shape[1], 3)
		if mode=='rgb':
			return N.copy(N.concatenate([patch.T[0].T, patch.T[1].T, patch.T[2].T]))
		elif mode=='rg' or mode=='rg_vs_b':
			return N.copy(N.concatenate([patch.T[0].T, patch.T[1].T]))	
		else:
			return N.copy(patch)



def get_random_patch_from_image(
	image, 
	patch_h, 
	patch_w,
	):
	bd_h, bd_w = patch_h*4, patch_w*4 # dist to border
	r = N.random.randint(bd_h, image.shape[0]-bd_h)
	c = N.random.randint(bd_w, image.shape[1]-bd_w)
	return N.copy(get_patch(image, r, c, patch_h, patch_w))



def get_patch(
	image, 
	off_row, 
	off_col, 
	h, 
	w
	):
	return image[off_row : off_row+h : 1, off_col : off_col+w : 1]




def get_avergae_variance_over_images(
	img_paths
	):
	'''calculate avg variance over ALL images'''
	avg_variance = 0
	for ipath in img_paths:
		img = read_image(img_paths[N.random.randint(0, len(img_paths))])
		avg_variance += N.var(img)
	avg_variance /= len(img_paths)
	print 'average_image_variance', avg_variance
	# 0.0589692271522
	return avg_variance









'''
https://www.cs.bris.ac.uk/~reinhard/papers/colourtransfer.pdf

RGB -> LMS
N.array([[0.3811, 0.5783, 0.0402],
		[0.1967, 0.7244, 0.0782],
		[0.0241, 0.1288, 0.8444],])

LMS -> RGB
N.array([[4.4679, -3.5873, 0.1193],
		[-1.2186, 2.3809, -0.1624],
		[0.0497, -0.2439, 1.2045],])


xyz = N.array([[.49, .31, .20], [.17691, .8124, .01063], [.0, .01, .99]]) * (1/17697.)
lms = N.array([[.8951, .2664, -.1614], [-.7502, 1.7135, .0367], [.0389, -.0685, 1.0296]])
>>> lms*xyz
array([[  2.47838052e-05,   4.66655365e-06,  -1.82403797e-06],
   [ -7.49945652e-06,   7.86600780e-05,   2.20444708e-08],
   [  0.00000000e+00,  -3.87071255e-08,   5.75975589e-05]])
	

http://en.wikipedia.org/wiki/LMS_color_space
http://en.wikipedia.org/wiki/CIE_1931_color_space
http://ssodelta.wordpress.com/tag/rgb-to-lms/
'''
def convert_patch_to_lms(
	patch,
	):
	xyzlms = N.array([[0.3811, 0.5783, 0.0402],
		[0.1967, 0.7244, 0.0782],
		[0.0241, 0.1288, 0.8444],])
	patch_lms = N.zeros(patch.shape)
	for row in xrange(patch.shape[0]):
		for col in xrange(patch.shape[1]):
			patch_lms[row, col] = N.dot(xyzlms, patch[row, col])
	return patch_lms#/255.

'''
Computational Colour Science using MATLAB, p 89.
http://books.google.de/books?id=zgeuoz_muCIC&pg=PA89&lpg=PA89&dq=ciecam97s+bradford+0.8951&source=web&ots=oJQ6ylBkOz&sig=91sZ-oHpaUHNivWBVyr6XyPm2B4&redir_esc=y#v=onepage&q=ciecam97s%20bradford%200.8951&f=false

tristimulus value X Y Z is normalized by Y
r =  .8951*X/Y +  .2664*Y/Y +  .1614*Z/Y
g = -.7502*X/Y + 1.7135*Y/Y +  .0367*Z/Y
b =  .0389*X/Y -  .0685*Y/Y + 1.0296*Z/Y


X = patch[row, col][0]
Y = patch[row, col][1]
Z = patch[row, col][2]

r =  .8951*X +  .2664*Y +  .1614*Z
g = -.7502*X + 1.7135*Y +  .0367*Z
b =  .0389*X -  .0685*Y + 1.0296*Z

patch[row, col] = N.array([r/Y, g/Y, b/Y])

'''

def convert_patch_to_rgb(
	patch,
	):
	rgb = N.array([[4.4679, -3.5873, 0.1193],
		[-1.2186, 2.3809, -0.1624],
		[0.0497, -0.2439, 1.2045],])
	patch_rgb = N.zeros(patch.shape)
	for row in xrange(patch.shape[0]):
		for col in xrange(patch.shape[1]):
			patch_rgb[row, col] = N.dot(rgb, patch[row, col])
			# X = patch[row, col][0]
			# Y = patch[row, col][1]
			# Z = patch[row, col][2]
			# r =  .8951*X +  .2664*Y +  .1614*Z
			# g = -.7502*X + 1.7135*Y +  .0367*Z
			# b =  .0389*X -  .0685*Y + 1.0296*Z
			# patch_rgb[row, col] = N.array([r, g, b])
	return patch_rgb





if __name__ == '__main__':
	patch_w = 8
	from ..base.images import get_paths_of_images, read_image
	paths = get_paths_of_images('../../da/data/Foliage', 'tif', verbose=False)
	img = read_image(
		paths[N.random.randint(0, len(paths))])
	patch = get_random_patch_from_image(img, patch_w, patch_w)

	print patch
	lms_patch = convert_patch_to_lms(patch)
	print lms_patch




