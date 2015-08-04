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


import sys
import numpy as N

'''
Some portions (writing of tiles) derived from from 
https://www2.informatik.uni-hamburg.de/~weber/code/KTimage.py
wriiten by C. Weber.
'''


def exporttiles(a,h,w,filename,x=None,y=None):
	'''
	Weight maxtrix or activity vector a is written in a file.
	A vector is written as a tile of size h*w.
	A matrix is written as a tile of tiles of size x*y
	'''	
	__exporttiles(a,h,w,filename,x,y,mode='lum')

def exporttiles_rg(a,h,w,filename,x=None,y=None):
	__exporttiles(a,h,w,filename,x,y,mode='rg')

def exporttiles_rg_vs_b(a,h,w,filename,x=None,y=None):
	__exporttiles(a,h,w,filename,x,y,mode='rg_vs_b')

def exporttiles_rgb(a,h,w,filename,x=None,y=None):
	__exporttiles(a,h,w,filename,x,y,mode='rgb')





def __write_tile(
	filename,
	**kwargs
	):
	a = kwargs['a']
	b = kwargs['b']
	frame = kwargs['frame']
	x = kwargs['x']
	y = kwargs['y']
	w = kwargs['w']
	h = kwargs['h']
	mode = kwargs['mode']

	def lum():
		_c = ''
		for i in range(height):
			for j in range(width):
				_c += chr(int(b[i][j]))
		return _c

	def rg():
		_c = ''
		for i in range(height):
			for j in range(width):
				_c += chr(int(b[i][j][0]))
				_c += chr(int(b[i][j][1]))
				_c += chr(int(128))
		return _c

	def rg_vs_b():
		_c = ''		
		for i in range(height):
			for j in range(width):
				_c += chr(int(b[i][j][0]))
				_c += chr(int(b[i][j][0]))
				_c += chr(int(b[i][j][1]))
		return _c			

	def rgb():
		_c = ''
		for i in range(height):
			for j in range(width):
				_c += chr(int(b[i][j][0]))
				_c += chr(int(b[i][j][1]))
				_c += chr(int(b[i][j][2]))		
		return _c

	if mode == 'lum':
		width, height = frame + y*(w+frame), frame + x*(h+frame)
		header = 'P5\n'
	else:
		width, height = frame + y*(w+frame), frame + x*(w+frame)
		header = 'P6\n'

	amax, amin = N.max(a), N.min(a)
	if mode != 'lum':
		if  N.max(a) != N.min(a):
			factor = 255.0 / (N.max(a) - N.min(a))
		else:
			factor = 0.0
		a = (a - N.min(a)) * factor


	f = open(filename, 'w')
	# write the header
	f.write(header)
	f.write('# highS: %.6f  lowS: %.6f\n' % (amax, amin))
	line = str(width) + " " + str(height) + "\n"
	f.write(line)
	f.write("255\n") # values range from 0 to 255
	f.close()
	# write the data
	if  N.max(b) != N.min(b):
		factor = 255.0 / (N.max(b) - N.min(b))
	else:
		factor = 0.0
	b = (b - N.min(b)) * factor
	if mode == 'lum':
		c = lum()
	elif mode == 'rg':
		c = rg()
	elif mode == 'rg_vs_b':
		c = rg_vs_b()
	elif mode == 'rgb':
		c = rgb()
	else:
		c = ''

	f = open(filename, 'r+b')
	f.seek(0,2)
	if sys.version[0] == "2":
		f.write(c)
	else:
		f.write(c.encode("ISO-8859-1"))
		f.close()



from ..base.receptivefield import stack_matrix
def __exporttiles(
	a,
	h,
	w,
	filename,
	x=None,
	y=None,
	mode='lum'
	):

	if x is None:
		a = N.reshape(a, (1, h*w))
		x, y, = 1, 1
		frame = 0
	else:
		frame = 1

	if mode == 'lum':
		xy, hw = N.shape(a)
		b = N.ones((frame + x*(h+frame), frame + y*(w+frame))) * 0.15
	else:
		if mode == 'rg' or mode == 'rg_vs_b':
			a = stack_matrix(a, w, y, mode='rg')
		elif mode == 'rgb':
			a = stack_matrix(a, w, y)
		xy, hw, chn = N.shape(a)
		b = N.ones((frame + x*(h+frame), frame + y*(w+frame), chn)) * .0333
	
	image_id = 0
	for xx in range(x):
		for yy in range(y):
			if image_id >= xy: 
				break
			if mode == 'lum':
				tile = N.reshape (a[image_id], (h, w))
				beginH, beginW = frame + xx*(h+frame), frame + yy*(w+frame)
				b[beginH : beginH+h, beginW : beginW+w] = tile
			elif mode == 'rgb':
				t_r = N.reshape(a[image_id].T[0], (w, w))
				t_g = N.reshape(a[image_id].T[1], (w, w))
				t_b = N.reshape(a[image_id].T[2], (w, w))
				beginH, beginW = frame + xx*(w+frame), frame + yy*(w+frame)
				b[beginH : beginH+w, beginW : beginW+w] = N.array([t_r, t_g, t_b]).T
			elif mode == 'rg' or mode == 'rg_vs_b':
				t_r = N.reshape(a[image_id].T[0], (w, w))
				t_g = N.reshape(a[image_id].T[1], (w, w))
				beginH, beginW = frame + xx*(w+frame), frame + yy*(w+frame)
				b[beginH : beginH+w, beginW : beginW+w] = N.array([t_r, t_g]).T
			image_id += 1

	__write_tile(filename, a=a, b=b, frame=frame, x=x, y=y, w=w, h=h, mode=mode)




