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

'''-------------------------------------------------------------------------'''

def __zero_abs_values_below_mean(a):
	a = N.copy(a)
	ab = N.abs(a)
	a[ab < N.mean(ab)] = 0
	return a


def num_relevant_weights(
	mode,
	rf,
	patch_w,
	channel_w):
	num = N.count_nonzero( __zero_abs_values_below_mean(rf) )
	return num


def chromaticy_of_rfvector(
	mode,
	rf,
	patch_w,
	channel_w):
	'''chromaticy - 'colorfulness', aka the
	difference of color components / channels of rf vector.
	grey value has value 0.'''
	def chroma(r, g, b):
		'''F = max(|gruen-rot|,|blau-rot|,|gruen-blau|)'''
		return N.max([N.max(N.abs(r-g)), N.max(N.abs(b-r)), N.max(N.abs(g-b))])
	if mode=='rgb':
		a = __zero_abs_values_below_mean(rf[0:channel_w])
		b = __zero_abs_values_below_mean(rf[channel_w:2*channel_w])
		c = __zero_abs_values_below_mean(rf[2*channel_w:3*channel_w])
		return chroma(a, b, c)
	elif mode=='rg' or mode=='rg_vs_b':
		a = __zero_abs_values_below_mean(rf[0:channel_w])
		b = __zero_abs_values_below_mean(rf[channel_w:2*channel_w])
		return N.max(N.abs(a-b))
	else:
		return 0


'''-------------------------------------------------------------------------'''


def value_of_rfvector_at(
	mode, 
	rf, 
	x, y, 
	patch_w, 
	channel_w
	):
	'''return value of rf vector by spatial x,y (matrix-) coordinates.'''
	x,y = N.round(x), N.round(y)
	if x < 0: x = 0
	if x > patch_w-1: x = patch_w-1
	if y < 0: y = 0
	if y > patch_w-1: y = patch_w-1
	idx = int(x)+int(y)*patch_w
	if mode=='rgb':
		return [rf[idx], rf[idx+channel_w], rf[idx+2*channel_w]]
	elif mode=='rg' or mode=='rg_vs_b':
		return [rf[idx], rf[idx+channel_w]]
	else:
		return [rf[idx]]


def color_direction_bias(
	mode, 
	rf, 
	channel_w
	):
	'''mean of each rf vector component'''
	if mode=='rgb':
		return [N.mean(rf[0:channel_w]), 
				N.mean(rf[channel_w:2*channel_w]), 
				N.mean(rf[2*channel_w:3*channel_w])]
	elif mode=='rg' or mode=='rg_vs_b':
		return [N.mean(rf[0:channel_w]),
				N.mean(rf[channel_w:2*channel_w])]
	else:
		return [N.mean(rf[0:channel_w])]


def arg_absmax_rfvector(
	mode,
	rf,
	patch_w,
	channel_w
	):
	'''indices of abs max value of each rf vector component'''
	def arg_vec2mat(i):	return [i % patch_w, i / patch_w]
	if mode=='rgb':
		a = N.argmax(N.abs(rf[0:channel_w]))
		b = N.argmax(N.abs(rf[channel_w:2*channel_w]))
		c = N.argmax(N.abs(rf[2*channel_w:3*channel_w]))
		return N.array([arg_vec2mat(a), arg_vec2mat(b), arg_vec2mat(c)])
	elif mode=='rg' or mode=='rg_vs_b':
		a = N.argmax(N.abs(rf[0:channel_w]))
		b = N.argmax(N.abs(rf[channel_w:2*channel_w]))
		return N.array([arg_vec2mat(a), arg_vec2mat(b)])
	elif mode=='lum':
		a = N.argmax(N.abs(rf))
		return N.array([arg_vec2mat(a)])


def is_close_to_zero(
	a,
	atol=1e-08,
	rtol=1e-05,
	verbose=False
	):
	allclose = N.allclose(a, N.zeros(a.shape), atol=atol)
	if verbose and allclose: 
		print '****Note: all RF weights are close to zero. atol =', atol
	return allclose


'''-------------------------------------------------------------------------'''


def convert_rfvector_to_rgbmatrix(
	a,
	patch_w,
	patch_h,
	mode,
	swap=True,
	flip=False,
	):
	if mode=='rgb':
		r, g, b = decompose_vector(a)
		org = compose_vector(r, g, b)
		org = org.reshape(patch_h,patch_w,3)
	elif mode=='rg':
		r, g = decompose_vector(a, mode=mode)
		org = compose_vector(r, g, N.zeros(r.shape[0]))
		org = org.reshape(patch_h,patch_w,3)
	elif mode=='rg_vs_b':
		r, g = decompose_vector(a, mode=mode)
		org = compose_vector(r, r, g)
		org = org.reshape(patch_h,patch_w,3)	
	elif mode=='lum':
		org = N.copy(a)
		org = org.reshape(patch_h,patch_w)
	if swap: org = N.swapaxes(org, 0, 1)
	if flip: org = N.flipud(org)
	return org


def stack_matrix(
	w,
	patch_w,
	patch_h,
	mode='rgb',
	):
	if mode=='lum':	return w
	elif mode=='rgb':	vis_ch = 3
	elif mode=='rg' or mode=='rg_vs_b':	vis_ch = 2
	m = N.zeros((patch_h**2, patch_w**2, vis_ch))
	if mode=='rgb':
		for row in xrange(0, w.shape[0]):
			r, g, b = decompose_vector(w[row])
			channel_magn = w[row].shape[0]/3
			for c in xrange(0, channel_magn):
				m[row, c] = (r[c], g[c], b[c])
	elif mode=='rg' or mode=='rg_vs_b':
		for row in xrange(0, w.shape[0]):
			r, g = decompose_vector(w[row], mode=mode)
			channel_magn = w[row].shape[0]/2
			for c in xrange(0, channel_magn):
				m[row, c] = (r[c], g[c])
	return N.copy(m)


def compose_vector(
	r,
	g,
	b=None,
	):
	channel_magn = r.shape[0]
	if b.all() != None:
		rf = N.zeros((channel_magn, 3))
		for c in xrange(0, channel_magn):
			rf[c] = (r[c], g[c], b[c])		
	else:
		rf = N.zeros((channel_magn, 2))
		for c in xrange(0, channel_magn):
			rf[c] = (r[c], g[c])
	return N.copy(rf)


def decompose_vector(
	a,
	mode='rgb'
	):
	if mode=='lum':
		return a
	elif mode=='rg_vs_b' or mode=='rg':
		channel_magn = a.shape[0]/2
		r = a[0:channel_magn:1]
		g = a[channel_magn:2*channel_magn:1]
		return r, g
	elif mode=='rgb':
		channel_magn = a.shape[0]/3
		r = a[0:channel_magn:1]
		g = a[channel_magn:2*channel_magn:1]
		b = a[2*channel_magn::1]
		return r, g, b


'''-------------------------------------------------------------------------'''


