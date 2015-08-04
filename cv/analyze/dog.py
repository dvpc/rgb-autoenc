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

def dog_a(
	p, 
	channel_w, 
	patch_w, 
	x):
	xc = x % channel_w
	px = xc % patch_w
	py = xc / patch_w
	'''	params: 0: mu_x, 1: mu_y, 2: r_c, 3: r_s, 
				4: k_s (k_c is implicitly fixed as 1)
				5: cdir_a, 
				6: bias_a, 
	'''
	dist = ((px-p[0])**2+(py-p[1])**2)**.5
	dogs = N.exp(-(dist/p[2])**2) - p[4] * N.exp(-(dist/p[3])**2)
	ret = p[6] + p[5] * dogs[0:channel_w]
	return ret


def dog_ab(
	p, 
	channel_w, 
	patch_w, 
	x):
	xc = x % channel_w
	px = xc % patch_w
	py = xc / patch_w
	'''	params: 0: mu_x, 1: mu_y, 2: r_c, 3: r_s, 
				4: k_s (k_c is implicitly fixed as 1)
				5: cdir_a, 6: cdir_b,
				7: bias_a, 8: bias_b,
	'''
	dist = ((px-p[0])**2+(py-p[1])**2)**.5
	dogs = N.exp(-(dist/p[2])**2) - p[4] * N.exp(-(dist/p[3])**2)
	ret_A = p[7] + p[5] * dogs[0:channel_w]
	ret_B = p[8] + p[6] * dogs[channel_w:2*channel_w]
	ret = N.concatenate([ ret_A, ret_B ])
	return ret


def dog_abc(
	p, 
	channel_w, 
	patch_w, 
	x):
	xc = x % channel_w
	px = xc % patch_w
	py = xc / patch_w
	'''	params: 0: mu_x, 1: mu_y, 2: r_c, 3: r_s, 
				4: k_s (k_c is implicitly fixed as 1)
				5: cdir_a, 6: cdir_b,  7: cdir_c
				8: bias_a, 9: bias_b, 10: bias_c
	'''
	dist = ((px-p[0])**2+(py-p[1])**2)**.5
	dogs = N.exp(-(dist/p[2])**2) - p[4] * N.exp(-(dist/p[3])**2)
	ret_A = p[8] + p[5] * dogs[0:channel_w]
	ret_B = p[9] + p[6] * dogs[channel_w:2*channel_w]
	ret_C = p[10]+ p[7] * dogs[2*channel_w:3*channel_w]
	ret = N.concatenate([ ret_A, ret_B, ret_C ])
	return ret


'''-------------------------------------------------------------------------'''


def volume_approx(
	p,
	):
	if p[2]<p[3]: r = N.abs(p[2]) # ret smaller 'center'
	else: 		  r = N.abs(p[3])
	return 2*N.pi*r


def reconstruct(
	p,
	mode, 
	channel_w, 
	patch_w, 
	shape):
	if 	 mode=='lum':		rec =   dog_a(p, channel_w, patch_w, *N.indices(shape))
	elif mode=='rg' or \
		 mode=='rg_vs_b': 	rec =  dog_ab(p, channel_w, patch_w, *N.indices(shape))
	elif mode=='rgb': 		rec = dog_abc(p, channel_w, patch_w, *N.indices(shape))
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
	''' params:  cx  cy  rc  rs  |  ks  cdir_a  cdir_b  cdir_c | bias_a  bias_b  bias_c
		lum:     x   x   x   x      x   x                        x 
		rg       x   x   x   x      x   x       x                x       x 
		rgb      x   x   x   x      x   x       x       x        x       x       x
	'''
	init_p = [patch_w/2, patch_w/2, 1.5, patch_w*.6]
	# bounds_p = [(1,patch_w-1),(1,patch_w-1),(1,patch_w/2),(1,patch_w/2)]
	if   mode=='rgb':
		init_p   += [1, 1, 1, 1]
		# bounds_p += [(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None)]
		f = dog_abc
	elif mode=='rg_vs_b' or mode=='rg':
		init_p   += [1, 1, 1]
		# bounds_p += [(None,None),(None,None),(None,None),(None,None),(None,None)]
		f = dog_ab
	else:
		init_p   += [1, 1]
		# bounds_p += [(None,None),(None,None),(None,None)]		
		f = dog_a
	bounds_p = []
	init_p += color_direction_bias(mode, rf, channel_w)
	constraints = (
		{'type': 'ineq', 'fun': lambda x:  x[0]},				# cx > 0
		{'type': 'ineq', 'fun': lambda x:  patch_w-1 - x[0]},	# cx < patch_w-1

		{'type': 'ineq', 'fun': lambda x:  x[1]},				# cy > 0
		{'type': 'ineq', 'fun': lambda x:  patch_w-1 - x[1]},	# cy < patch_w-1


		{'type': 'ineq', 'fun': lambda x:  x[2] - .5},			# rc > .75
		{'type': 'ineq', 'fun': lambda x:  x[3] - .5},			# rc > .75

		{'type': 'ineq', 'fun': lambda x:  x[2] - .2*x[3]}, 	# rc*0.75 < rs
		{'type': 'ineq', 'fun': lambda x:  x[3] - 1.8*x[2]},	# rs*1.25 < rc

		{'type': 'ineq', 'fun': lambda x:  .3 - x[4]},			# k 
		)

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
			debug_write_fit(best_p, f, rf, channel_w, patch_w, mode, info, model='dog', path=debug_output_path)
	else:
		debug_func = lambda x,xy,y,z:x		
	best_p = bestfit_skel(perm_p_f=perm, fit_f=fit_func, debug_f=debug_func,
						  init_p=init_p, num_attempts=num_attempts, return_error=return_error)
	return best_p



if __name__ == '__main__':

	# fname = 'Wrgc_8x8_lum_k6e-05_lr0.08_normL0.5.rgc'
	# fname = 'Wrgc_8x14_rg_clip_k6e-06_lr0.08_normL0.5.rgc'
	fname = 'W_8x14_rg_vs_b_clip_k6e-06_lr0.08_normL0.5.rgc'
	# fname = 'Wrgc_8x14_rgb_k6e-05_lr0.08_normL0.5soft.rgc'

	args_outdir = '../../rgc_working_dir/'
	args_file = args_outdir +'out/'+ fname #'../../da/__out/' + fname

	# keys = [195,156,123,99,83,57,66,14]
	# keys = [145,129,112,24,23,12,6]

	# exclude = [194,190,185,176,166,147,126,111,108,84,35,1]

	# keys = [190,167,126,111,1]



	keys = [166]

	from weightmatrix import load
	W_args = load(args_file, dont_load_matrix=True, verbose=False)
	W = load(args_file, dont_load_matrix=False)
	patch_w, W_mode = W_args[0], W_args[2]
	channel_w = patch_w**2

	for key in keys:
		rf = W[key]
		print bestfit(W_mode, rf, channel_w, patch_w, num_attempts=20, debug=True, key=key, maxiter=5000)

	

