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

'''
http://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function

f(x,y) 		= A * exp [ - ( a *(x-mu_x)^2 + 2*b*(x-mu_x)*(y-mu_y) + c(y-mu_y)^2 ) ]
a 			= cos(theta)^2/2/sigma_x^2  + sin(theta)^2/2/sigma_y^2
b 			= -sin(2*theta)/4/sigma_x^2 + sin(2*theta)/4/sigma_y^2
c 			= sin(theta)^2/2/sigma_x^2  + cos(theta)^2/2/sigma_y^2

A 			~ peak height / amplitude
mu_xy 		~ center of blob
sigma_xy	~ xy spread of blob
theta 		~ angle to rotate the blob

====>

params p:
0 mu_x
1 mu_y
2 sigma_x
3 sigma_y
4 c_to_s_ratio
5 theta (rotation)
6 k_s (k_c is implicitly fixed as 1)
7 cdir_a,
8 bias_a

def gauss2d(scale=1):
	sigma_x = p[2]^2*scale
	sigma_y = p[3]^2*scale
	a =  cos(p[5])^2/2/sigma_x + sin(p[5])^2/2/sigma_y
	b = -sin(2*p[5])/4/sigma_x + sin(2*p[5])/4/sigma_y
	c =  sin(p[5])^2/2/sigma_x + cos(p[5])^2/2/sigma_y
	return N.exp( - ( a*(px-p[0])**2 + 2*b*(px-p[0])*(py-p[1]) + c*(py-p[1])**2 ) )


dogs = gauss2d(scale=1) - p[6] * gauss2d(scale=p[4])
ret = p[8] + p[7] * dogs[0:channel_w]

'''

def edog_a(
	p, 
	channel_w, 
	patch_w, 
	x):
	xc = x % channel_w
	px = xc % patch_w
	py = xc / patch_w
	'''params: 0: mu_x  1: mu_y  2: sigma_x  3: sigma_y  4: c_to_s_ratio
			   5: theta (rotation) 6: k_s (k_c is implicitly fixed as 1)
			   7: cdir_a, 8: bias_a
	'''
	def gauss2d(scale=1):
		sigma_x = p[2]**2*scale
		sigma_y = p[3]**2*scale
		a =  N.cos(p[5])**2/2/sigma_x + N.sin(p[5])**2/2/sigma_y
		b = -N.sin(2*p[5])/4/sigma_x + N.sin(2*p[5])/4/sigma_y
		c =  N.sin(p[5])**2/2/sigma_x + N.cos(p[5])**2/2/sigma_y
		return N.exp( - ( a *(px-p[0])**2 + 2*b*(px-p[0])*(py-p[1]) + c*(py-p[1])**2 ) )
	dogs = gauss2d(scale=1) - p[6] * gauss2d(scale=p[4])
	ret = p[8] + p[7] * dogs[0:channel_w]
	return ret


def edog_ab(
	p, 
	channel_w, 
	patch_w, 
	x):
	xc = x % channel_w
	px = xc % patch_w
	py = xc / patch_w
	'''params: 0: mu_x  1: mu_y  2: sigma_x  3: sigma_y  4: c_to_s_ratio
			   5: theta (rotation) 6: k_s (k_c is implicitly fixed as 1)
			   7: cdir_a, 8:  cdir_b,
			   9: bias_a, 10: bias_b,
	'''
	def gauss2d(scale=1):
		sigma_x = p[2]**2*scale
		sigma_y = p[3]**2*scale
		a =  N.cos(p[5])**2/2/sigma_x + N.sin(p[5])**2/2/sigma_y
		b = -N.sin(2*p[5])/4/sigma_x + N.sin(2*p[5])/4/sigma_y
		c =  N.sin(p[5])**2/2/sigma_x + N.cos(p[5])**2/2/sigma_y
		return N.exp( - ( a *(px-p[0])**2 + 2*b*(px-p[0])*(py-p[1]) + c*(py-p[1])**2 ) )
	dogs = gauss2d(scale=1) - p[6] * gauss2d(scale=p[4])
	ret_A = p[9] + p[7] * dogs[0:channel_w]
	ret_B = p[10]+ p[8] * dogs[channel_w:2*channel_w]
	ret = N.concatenate([ ret_A, ret_B ])
	return ret


def edog_abc(
	p, 
	channel_w, 
	patch_w, 
	x):
	xc = x % channel_w
	px = xc % patch_w
	py = xc / patch_w
	'''params: 0: mu_x  1: mu_y  2: sigma_x  3: sigma_y  4: c_to_s_ratio
			   5: theta (rotation) 6: k_s (k_c is implicitly fixed as 1)
			   7: cdir_a, 8: cdir_b,  9: cdir_c
			   10: bias_a, 11: bias_b, 12: bias_c
	'''
	def gauss2d(scale=1):
		sigma_x = p[2]**2*scale
		sigma_y = p[3]**2*scale
		a =  N.cos(p[5])**2/2/sigma_x + N.sin(p[5])**2/2/sigma_y
		b = -N.sin(2*p[5])/4/sigma_x + N.sin(2*p[5])/4/sigma_y
		c =  N.sin(p[5])**2/2/sigma_x + N.cos(p[5])**2/2/sigma_y
		return N.exp( - ( a *(px-p[0])**2 + 2*b*(px-p[0])*(py-p[1]) + c*(py-p[1])**2 ) )
	dogs = gauss2d(scale=1) - p[6] * gauss2d(scale=p[4])
	ret_A = p[10] + p[7] * dogs[0:channel_w]
	ret_B = p[11] + p[8] * dogs[channel_w:2*channel_w]
	ret_C = p[12] + p[9] * dogs[2*channel_w:3*channel_w]
	ret = N.concatenate([ ret_A, ret_B, ret_C ])
	return ret








'''-------------------------------------------------------------------------'''


def volume_approx(
	p,
	):
	if p[4]>1: 	r = p[2]*p[3]*p[4] # 4: c_to_s_ratio
	else: 		r = p[2]*p[3]
	return N.abs(r)


def reconstruct(
	p,
	mode, 
	channel_w, 
	patch_w, 
	shape):
	if 	 mode=='lum':		rec =   edog_a(p, channel_w, patch_w, *N.indices(shape))
	elif mode=='rg' or \
		 mode=='rg_vs_b': 	rec =  edog_ab(p, channel_w, patch_w, *N.indices(shape))
	elif mode=='rgb': 		rec = edog_abc(p, channel_w, patch_w, *N.indices(shape))
	return rec



from ..base.receptivefield import color_direction_bias
from .util import permutate_mu
from .util import fit_slsqp
from .util import bestfit_skel
def bestfit(
	mode, 
	rf, 
	channel_w, 
	patch_w, 
	num_attempts=10,
	maxiter=10000,
	debug=False,
	key=None,
	return_error=False,
	debug_output_path='../'
	):
	constraints = (
		{'type': 'ineq', 'fun': lambda x:  x[0]},				# cx > 0
		{'type': 'ineq', 'fun': lambda x:  patch_w-1 - x[0]},	# cx < patch_w-1

		{'type': 'ineq', 'fun': lambda x:  x[1]},				# cy > 0
		{'type': 'ineq', 'fun': lambda x:  patch_w-1 - x[1]},	# cy < patch_w-1

		{'type': 'ineq', 'fun': lambda x:  x[2]-.65*x[3]},		# sigma_x > 0.7 sigma_y
		{'type': 'ineq', 'fun': lambda x:  2.25*x[3]-x[2]},		# sigma_x < 1.3 sigma_y
		)

	if N.abs(N.min(rf)) > N.abs(N.max(rf)):
		k_s = -.5
		constraints += (
		{'type': 'ineq', 'fun': lambda x:  x[2] - .15},			# sigma > .3
		{'type': 'ineq', 'fun': lambda x:  x[3] - .15},			# sigma > .3	

		{'type': 'ineq', 'fun': lambda x:  1.2 - x[4]},
		)
		
	else:
		k_s = .5
		constraints += (
		{'type': 'ineq', 'fun': lambda x:  x[2] - .25},			# sigma > .3
		{'type': 'ineq', 'fun': lambda x:  x[3] - .25},			# sigma > .3			

		{'type': 'ineq', 'fun': lambda x:  x[4] - 1.2},
		)


	'''        0              1              2             3             4             5         6       7       8       9          10      11      12       '''
	'''params: mu_x           mu_y           sigma_x       sigma_y       c_to_s_ratio  theta     k_s  |  cdir_a  cdir_b  cdir_c  |  bias_a  bias_b  bias_c  |'''
	init_p   = [patch_w/2,    patch_w/2,     1,            1,            1.,           N.pi/4.,  k_s]
	if   mode=='rgb':
		init_p   += [1, 1, 1]
		f = edog_abc
	elif mode=='rg_vs_b' or mode=='rg':
		init_p   += [1, 1]
		f = edog_ab
	else:
		init_p   += [1]
		f = edog_a
	bounds_p = []
	init_p += color_direction_bias(mode, rf, channel_w)

	def perm(n, init_p):
		return permutate_mu(n, init_p, patch_w, channel_w, mode, rf)
	def fit_func(p):
		return fit_slsqp(f, rf, p, bounds_p, channel_w, patch_w, maxiter=maxiter, constraints=constraints)
	if debug:
		from util import debug_write_fit
		def debug_func(idx, best_p, best_fun, msg):
			print 'best attempt: ', idx, '- fun:', best_fun, '\n', best_p
			if key is None: info = str(msg)
			else: 			info = str(key)
			debug_write_fit(best_p, f, rf, channel_w, patch_w, mode, info, model='edog', path=debug_output_path)
	else:
		debug_func = lambda x,xy,y,z:x
	'''http://docs.scipy.org/doc/numpy/reference/generated/numpy.seterr.html'''
	N.seterr(all='ignore')
	best_p = bestfit_skel(perm_p_f=perm, fit_f=fit_func, debug_f=debug_func,
						  init_p=init_p, num_attempts=num_attempts, return_error=return_error)
	N.seterr(all='warn')
	return best_p



















if __name__ == '__main__':


	# fname = 'Wrgc_8x8_lum_k6e-05_lr0.08_normL0.5.rgc'
	# fname = 'Wrgc_8x14_rg_clip_k6e-06_lr0.08_normL0.5.rgc'
	# fname = 'W_8x14_rg_vs_b_clip_k6e-06_lr0.08_normL0.5.rgc'
	# fname = 'Wrgc_8x14_rgb_k6e-05_lr0.08_normL0.5soft.rgc'
	# fname = 'W(8x24)_rgb__k1e-05_p0.5_lr0.055__clip.map'
	fname = 'W(8x24)_rgb__k1e-05_p0.5_lr0.05__clip_p_corrupt1.0.map'
	
	# args_file = '../../da/__out/' + fname
	args_file = '../../rgc_working_dir/goett/' + fname
	args_outdir = '../../rgc_working_dir/'	
	
	
	# keys = [194,190,185,176,166,147,126,111,108,84,35,1]
	keys = [524]

	from weightmatrix import load_2 as load
	W_args = load(args_file, dont_load_matrix=True, verbose=False)
	W = load(args_file, dont_load_matrix=False)
	# patch_w, W_mode = W_args[0], W_args[2]
	patch_w, W_mode = W_args['patch_w'], W_args['mode']
	channel_w = patch_w**2

	for key in keys:
		rf = W[key]
		print bestfit(W_mode, rf, channel_w, patch_w, num_attempts=15, debug=True, key=key, maxiter=5000)


	

