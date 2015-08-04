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


def prune_map(
	args_file,
	args_odir,
	sort_row_then_col=True,
	normalize_weights=False
	):
	import numpy as N


	'''loading precomputed fits'''
	from util import depickle_fits
	res = depickle_fits(args_file, suffix='cfits')


	'''retrieving weightmatrix from fits dic'''
	W = res['map']['W']
	dic_W = res['map']
	patch_w = dic_W['vis']

	print N.max(W), N.min(W)

	'''normalize weights'''
	if normalize_weights:
		W = W / N.max(N.abs(W))
	
	print N.max(W), N.min(W)

	'''get fit and cluster data'''
	fits = res['fits']
	cl = res['dic_cluster']


	num_channel = cl['num_clusters']


	from ..base.receptivefield import is_close_to_zero
	'''move RF ids in an appropiate data struc, ignore zeros'''
	num_zeros = W.shape[0] - len(fits.keys())
	clusters_by_index = [[] for c in [None]*num_channel]	
	keys = sorted(res['fits'].keys())
	for i, key in enumerate(keys):
		rf = W[key]
		abs_rf = N.abs(rf)
		abs_max = N.max( abs_rf )
		abs_min = N.min( abs_rf )
		value_spectrum = abs_max - abs_min
		if value_spectrum > 0.2 and \
		not is_close_to_zero(rf, verbose=False, atol=1e-02):
			cl = res['fits'][key]['cl']
			clusters_by_index[cl].append(key)



	'''vectorize fit center coords clusterwise'''
	sorted_clusters_by_index = [[] for s in [None]*num_channel]
	for c in xrange(0,len(clusters_by_index)):
		cl_ids = clusters_by_index[c]
		dtype = [('coord', float), ('id', int)]
		values = []
		for i in xrange(0, len(cl_ids)):
			p = fits[cl_ids[i]]['p']
			p01 = (int(N.round(p[0])), int(N.round(p[1])))
			if sort_row_then_col:
				'''vectorize col wise'''
				values.append( (p01[1]+p01[0]*patch_w, cl_ids[i]) )
			else:
				'''vectorize row wise'''
				values.append( (p01[0]+p01[1]*patch_w, cl_ids[i]) )
		vectorized_coords = N.array(values, dtype)
		for pair in N.sort(vectorized_coords):
			sorted_clusters_by_index[c].append( pair[1] )


	not_zero = W.shape[0] - num_zeros
	'''new build pruned map'''
	if dic_W['mode'] == 'rgb': n_ch = 3
	elif dic_W['mode'] == 'rg_vs_b': n_ch = 2
	elif dic_W['mode'] == 'rg': n_ch = 2
	else: n_ch = 1
	pruned_patch_h = int(N.ceil(not_zero**.5))+1#2
	pruned_W_shape = (pruned_patch_h**2, patch_w**2*n_ch)
	rec_W = N.zeros(pruned_W_shape)
	rec_W_channel_dim = [(0,0) for dim in [None]*num_channel]
	rec_W_index = 0
	# rec_W_channel_coords = []
	for c in xrange(0,len(sorted_clusters_by_index)):
		cl_ids = sorted_clusters_by_index[c]
		rec_W_channel_dim[c] = (rec_W_index, rec_W_index+len(cl_ids)-1)
		to_nextrow = rec_W_index % pruned_patch_h
		if to_nextrow != 0:
			rec_W_index += pruned_patch_h - to_nextrow
		'''coords: store fitted center coords for each id'''
		# rec_W_channel_coords.append([])
		for i in xrange(0, len(cl_ids)):
			p = fits[cl_ids[i]]['p']
			rec_W[rec_W_index] = W[cl_ids[i]]
			rec_W_index += 1
			# pair = (int(N.round(p[0])), int(N.round(p[1])))
			# rec_W_channel_coords[c].append(pair)
			


	'''write pruned map'''
	d = res['map']
	W_rec_args = {
		'hid': pruned_patch_h,
		'vis': d['vis'],
		'mode': d['mode'],
		'k': d['k'],
		'p': d['p'], 
		'lr': d['lr'], 
		'clip': d['clip'],
		'version': d['version'], 
		'epochs_done': d['epochs_done'], 
		'chdim': rec_W_channel_dim,
		# 'chcoords': rec_W_channel_coords,
	}
	from pprint import pprint as pp
	print pp(W_rec_args)

	from ..base import rgc_filename_str
	str_mapfile = rgc_filename_str(
		mode=d['mode'], clip=d['clip'], 
		k=d['k'], p=d['p'], lr=d['lr'], 
		vis=d['vis'], hid=d['hid'],
		)


	import os
	mod_odir = os.path.join(args_odir, '../')
	from ..base import make_working_dir_sub
	work_dir = make_working_dir_sub(mod_odir, 'pr')
	from ..base import make_filename
	mapfile = make_filename(args_file, 'pruned_and_sorted_'+str_mapfile, '.map', odir=work_dir)

	from ..base.weightmatrix import save_2
	save_2(filepath=mapfile+'.map', 
		W=rec_W, 
		W_args=W_rec_args, 
		version=2, 
		verbose=True)


	imagefile = make_filename(args_file, 'pruned_and_sorted_'+str_mapfile, 
		'.png', odir=work_dir)

	'''write reconstructed map'''
	from ..base.plots import save_w_as_image
	save_w_as_image(X=rec_W, 
		in_w=dic_W['vis'], in_h=dic_W['vis'],
		out_w=pruned_patch_h, out_h=pruned_patch_h,
		outfile=imagefile+'.png',
		mode=dic_W['mode'],
		dpi=288)
	



if __name__ == '__main__':
	pass
