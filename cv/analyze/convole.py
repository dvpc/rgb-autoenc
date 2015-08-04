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


'''
https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.convolve.html
http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.filters.convolve.html

http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html

https://gist.github.com/thearn/5424195
'''

def convole_image_with_filter(
	args_file,
	input_image,
	odir = None,
	lum_channels = False,
	odir_lum_channels = False,
	odir_nosub = False,
	):
	from util import depickle_fits
	res = depickle_fits(args_file, suffix='kern')
	kernels = res['filters']

	import os
	if odir == None: 
		odir = os.path.dirname(args_file)
	if not odir_nosub:
		mod_odir = os.path.join(odir, '../')
		from ..base import make_working_dir_sub
		work_dir = make_working_dir_sub(mod_odir, 'conv')
	else:
		work_dir = odir

	if input_image == None:
		from scipy import misc
		img = misc.lena()*-1
	else:
		from ..base.images import read_image
		img = read_image(input_image, verbose=True)


	import numpy as N
	from scipy import signal

	def filter_rgb(image, kernel):
		r = signal.fftconvolve(image.T[0].T, kernel.T[0].T, mode='same')*.5 # full, same, valid
		r += signal.fftconvolve(image.T[0].T, kernel.T[0], mode='same')*.5
		g = signal.fftconvolve(image.T[1].T, kernel.T[1].T, mode='same')*.5
		g += signal.fftconvolve(image.T[1].T, kernel.T[1], mode='same')*.5
		b = signal.fftconvolve(image.T[2].T, kernel.T[2].T, mode='same')*.5
		b += signal.fftconvolve(image.T[2].T, kernel.T[2], mode='same')*.5

		# r = signal.convolve2d(image.T[0].T, kernel.T[0].T, boundary='symm', mode='same')
		# g = signal.convolve2d(image.T[1].T, kernel.T[1].T, boundary='symm', mode='same')
		# b = signal.convolve2d(image.T[2].T, kernel.T[2].T, boundary='symm', mode='same')
		return r, g, b

	def write_image(arr, name, suffix='.png', mode=None):
		from ..base.plots import colormap_for_mode
		if mode==None:
			cmap = colormap_for_mode(res['mode'])
		else:
			cmap = colormap_for_mode(mode)
		from ..base import make_filename
		imagefile = make_filename(args_file, name, suffix, work_dir)
		import matplotlib.image as pltimg
		pltimg.imsave(imagefile, arr, dpi=144, cmap=cmap)

	def wr_rgb(a,b,c,name):
		from ..base.plots import normalize_color
		print name, '   \t\t', N.max(N.dstack([a, b, c]))-N.min(N.dstack([a, b, c]))
		write_image(normalize_color(N.dstack([a, b, c])), name)

	def wr_lum(a,name,invert=False):
		from ..base.plots import normalize_color
		if invert:
			write_image(normalize_color(a)*-1, name, mode='lum')
		else:
			write_image(normalize_color(a), name, mode='lum')



	if len(img.shape) > 2:

		assert len(kernels) >= 5
		ch0_r, ch0_g, ch0_b = filter_rgb(img, kernels[0]-N.mean(kernels[0], axis=0))
		ch1_r, ch1_g, ch1_b = filter_rgb(img, kernels[1]-N.mean(kernels[1], axis=0))
		ch2_r, ch2_g, ch2_b = filter_rgb(img, kernels[2]-N.mean(kernels[2], axis=0))
		ch3_r, ch3_g, ch3_b = filter_rgb(img, kernels[3]-N.mean(kernels[3], axis=0))
		ch4_r, ch4_g, ch4_b = filter_rgb(img, kernels[4]-N.mean(kernels[4], axis=0))

		if len(kernels) == 5:
			wr_rgb(ch1_r, ch1_g, ch1_b, '0_red')
			wr_rgb(ch0_r, ch0_g, ch0_b, '1_green')
			wr_rgb(ch2_r, ch2_g, ch2_b, '2_blue')
			wr_rgb(ch3_r, ch3_g, ch3_b, '3_white')
			wr_rgb(ch4_r, ch4_g, ch4_b, '4_black')
	
			wr_rgb(ch1_r-ch0_r, ch1_g-ch0_g, ch1_b-ch0_b, '5_red_green')
			wr_rgb(ch0_r-ch1_r, ch0_g-ch1_g, ch0_b-ch1_b, '6_green_red')
			wr_rgb(ch2_r-(ch0_r+ch1_r)/2., ch2_g-(ch0_g+ch1_g)/2., ch2_b-(ch0_b+ch1_b)/2., '7_blue_yellow')

		if len(kernels) == 6:
			ch5_r, ch5_g, ch5_b = filter_rgb(img, kernels[5]-N.mean(kernels[5], axis=0))

			if not odir_lum_channels:
				if not lum_channels:
					wr_rgb(ch0_r, ch0_g, ch0_b, '0_red_on')
					wr_rgb(ch1_r, ch1_g, ch1_b, '1_green_on')
					wr_rgb(ch2_r, ch2_g, ch2_b, '2_blue_on')
					wr_rgb(ch4_r, ch4_g, ch4_b, '3_red_off')
					wr_rgb(ch3_r, ch3_g, ch3_b, '4_green_off')
					wr_rgb(ch5_r, ch5_g, ch5_b, '5_blue_off')
					wr_rgb(ch0_r+ch4_r, ch0_g+ch4_g, ch0_b+ch4_b, '6_red_on_green_off')
					wr_rgb(ch3_r-ch1_r, ch3_g-ch1_g, ch3_b-ch1_b, '6_green_on_red_off')

				else:
					wr_rgb(ch0_r, ch0_g, ch0_b, '0_luminosity_ON')
					wr_rgb(ch1_r, ch1_g, ch1_b, '1_luminosity_OFF')
					wr_rgb(ch2_r, ch2_g, ch2_b, '2_red_on')
					wr_rgb(ch4_r, ch4_g, ch4_b, '3_green_on')
					wr_rgb(ch3_r, ch3_g, ch3_b, '4_red_off')
					wr_rgb(ch5_r, ch5_g, ch5_b, '5_green_off')
					wr_rgb(ch2_r+ch5_r, ch2_g+ch5_g, ch2_b+ch5_b, '6_red_on_green_off')
					wr_rgb(ch4_r-ch3_r, ch4_g-ch3_g, ch4_b-ch3_b, '6_green_on_red_off')

			else:
				if not lum_channels:
					write_image((ch0_r+ch0_g+ch0_b)*-1, 'lum_0_red_on', mode='lum')
					write_image((ch1_r+ch1_g+ch1_b)*-1, 'lum_1_green_on', mode='lum')
					write_image((ch2_r+ch2_g+ch2_b)*-1, 'lum_2_blue_on', mode='lum')
					write_image((ch4_r+ch4_g+ch4_b)*-1, 'lum_3_red_off', mode='lum')
					write_image((ch3_r+ch3_g+ch3_b)*-1, 'lum_4_green_off', mode='lum')
					write_image((ch5_r+ch5_g+ch5_b)*-1, 'lum_5_blue_off', mode='lum')
					write_image((ch0_r+ch4_r+ch0_g+ch4_g+ch0_b+ch4_b)*-1, 'lum_6_red_on_green_off', mode='lum')
					write_image((ch3_r+ch1_r+ch3_g+ch1_g+ch3_b+ch1_b)*-1, 'lum_6_green_on_red_off', mode='lum')

				else:
					write_image((ch0_r+ch0_g+ch0_b)*-1, 'lum_0_luminosity_ON', mode='lum')
					write_image((ch1_r+ch1_g+ch1_b)*-1, 'lum_1_luminosity_OFF', mode='lum')
					write_image((ch2_r+ch2_g+ch2_b)*-1, 'lum_2_red_on', mode='lum')
					write_image((ch4_r+ch4_g+ch4_b)*-1, 'lum_3_green_on', mode='lum')
					write_image((ch3_r+ch3_g+ch3_b)*-1, 'lum_4_red_off', mode='lum')
					write_image((ch5_r+ch5_g+ch5_b)*-1, 'lum_5_green_off', mode='lum')
					write_image((ch2_r+ch5_r+ch2_g+ch5_g+ch2_b+ch5_b)*-1, 'lum_6_red_on_green_off', mode='lum')
					write_image((ch4_r+ch3_r+ch4_g+ch3_g+ch4_b+ch3_b)*-1, 'lum_6_green_on_red_off', mode='lum')				

	else:
		on_filterd = signal.convolve2d(img, kernels[0], boundary='symm', mode='same')
		off_filterd = signal.convolve2d(img, kernels[1], boundary='symm', mode='same')
		img_filterd = on_filterd - off_filterd

		write_image(img_filterd*-1, 'lum_convole')
		write_image(on_filterd, '0_lum_on')
		write_image(off_filterd*-1, '1_lum_off')


	# import os
	# working_dir = os.path.split(args_file)[0]
	# original = os.path.join(working_dir, 'org.png')
	# if not os.path.exists(original):
	write_image(img, 'org')

