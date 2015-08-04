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


def init_weight_matrix(
	mode,
	vis, 
	hid, 
	init_type_local=True,
	sparse=False,
	stddev=.03
	):
	from ..base import num_channel_of
	num_visible_channels = num_channel_of(mode)
	h_lin = hid*hid
	w_lin = vis*vis
	if sparse:
		_W = N.zeros((h_lin, w_lin*num_visible_channels))
		for k in xrange(_W.shape[0]):
			'''Sparse-3 Initialization'''
			_W[k,N.random.randint(0, w_lin)] = 0.456463462775
			_W[k,N.random.randint(0, w_lin)] = -1.43515478736
			_W[k,N.random.randint(0, w_lin)] = N.tanh(N.tanh(N.random.normal(0, 1, (1))))
	else:
		if init_type_local:
			_W = N.zeros((h_lin, w_lin*num_visible_channels))
			for k in xrange(_W.shape[0]):
				if num_visible_channels == 3:
					_W[k,0:w_lin]         = N.random.normal(0, stddev, (w_lin))
					_W[k,w_lin:2*w_lin]   = N.random.normal(0, stddev, (w_lin))
			 		_W[k,2*w_lin:3*w_lin] = N.random.normal(0, stddev, (w_lin))
			 	elif num_visible_channels == 2:
					_W[k,0:w_lin]         = N.random.normal(0, stddev, (w_lin))
					_W[k,w_lin:2*w_lin]   = N.random.normal(0, stddev, (w_lin))			 		
			 	elif num_visible_channels == 1:
			 		_W[k]                 = N.random.normal(0, stddev, (w_lin))
			 	else:
			 		print '**** Error: num_visible_channels < 1 or > 3 !'
			 		exit()
		else:
			_W = N.random.normal(0, .003, (h_lin, w_lin*num_visible_channels))
	return _W



'''-------------------------------------------------------------------------'''
import cPickle


def save_2(
	filepath='./W.rgc',
	W=None,
	W_args={},
	version=0,
	verbose=False
	):
	# print
	# for name, value in W_args.items():
	# 	print '{0} = {1}'.format(name, value)	
	# print
	dfile = open(filepath, 'wb')
	W_args['version'] = version
	cPickle.dump(W_args, dfile, -1)
	cPickle.dump(W, dfile, -1)
	if (verbose):
		print 'file:', filepath, 'written.'




def load_2(
	filepath, 
	dont_load_matrix=None, 
	verbose=False
	):
	import re
	match = re.match(r'.*\.map$', filepath)
	if match is None:
		print 'a pickled map (weight matrix) is expected (filename.map).\nexiting.'
		exit()
	lfile = open(filepath)
	W_args = cPickle.load(lfile)
	W = None ; strWshape = ''
	if dont_load_matrix is None or dont_load_matrix is False: 
	   W = cPickle.load(lfile)
	   strWshape = '\nshape: ' + ' '*(6-len(str('shape')))+'\t' + str(W.shape)
	lfile.close()
	if verbose:
		print filepath + strWshape
		keys = W_args.keys()
		for key in sorted(keys):
			print key+':', ' '*(6-len(str(key)))+'\t', W_args[key]

	if dont_load_matrix is None:
		return W_args, W
	elif dont_load_matrix is False:
		return W
	elif dont_load_matrix is True:
		return W_args
	


def load_2_as_dict(
	argsfile, 
	verbose=False,
	dont_load_matrix=None,
	):
	from ..base import extract_filename_without_suffix
	rval = load_2(argsfile, dont_load_matrix=dont_load_matrix, verbose=verbose)
	if dont_load_matrix is None or dont_load_matrix is False:
		W_args, W = rval
	elif dont_load_matrix is True:
		W_args, W = rval, None
	return dict(W_args.items() + {
		'W':W, 
		'file_name':extract_filename_without_suffix(argsfile)
		}.items())



'''-------------------------------------------------------------------------'''



def get_keys_of_W_in_bounds_byargs(
	args, 
	W
	):
	keys = get_keys_of_W_in_bounds(args.begin, args.num_iter, W.shape[0])
	return keys



def get_keys_of_W_in_bounds(
	begin,
	niter,
	W_shape_0,
	):
	'''integrity check of begin and niter'''
	if begin is None:				begin = 0
	if niter is None:				niter = W_shape_0
	if begin < 0:					begin = 0	
	if niter > W_shape_0:			niter = W_shape_0 - begin
	if niter < 1:					niter = 1
	if begin + niter > W_shape_0:	begin = W_shape_0 - niter
	keys = range(begin, begin+niter)
	return keys



from receptivefield import is_close_to_zero
def map_W_by_keys(
	W, 
	keys,
	f=lambda x,y:x
	):
	for i in xrange(0, len(keys)):
		rf = W[keys[i]]
		if is_close_to_zero(rf, verbose=False): continue
		f(keys[i], rf)

