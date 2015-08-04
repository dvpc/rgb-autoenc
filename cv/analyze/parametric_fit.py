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


def fit_debug(
	map_file,
	begin,
	num_iter,
	model,
	nfits,
	maxitr,
	ncpu,
	odir
	):
	from ..base.weightmatrix import load_2_as_dict
	wld_args = load_2_as_dict(map_file, dont_load_matrix=None, verbose=True)

	from ..base.weightmatrix import get_keys_of_W_in_bounds
	keys = get_keys_of_W_in_bounds(begin, num_iter, wld_args['W'].shape[0])

	from ..base import make_working_dir_sub
	work_dir = make_working_dir_sub(odir, 'dbg')

	__fit_map(wld_args, keys, model, nfits, maxitr, ncpu, 
		debug=True,
		debug_output_path=work_dir)


def fit_map_pickle_result(
	map_file,
	begin,
	num_iter,
	model,
	nfits,
	maxitr,
	ncpu,
	odir
	):
	from ..base.weightmatrix import load_2_as_dict
	wld_args = load_2_as_dict(map_file, dont_load_matrix=None, verbose=True)

	from ..base.weightmatrix import get_keys_of_W_in_bounds
	keys = get_keys_of_W_in_bounds(begin, num_iter, wld_args['W'].shape[0])

	fits = __fit_map(wld_args, keys, model, nfits, maxitr, ncpu)

	result = {'model':model, 
			  'fits':fits,
			  'map':wld_args
			  }	

	# from ..base import make_working_dir_sub
	# work_dir = make_working_dir_sub(odir, 'fits')

	from ..base import make_filename
	writefilename = make_filename(map_file,
		'paramfits_rgc_'+model+'_'+'('+str(keys[0])+','+str(keys[-1])+')',
		'.fits', odir=odir)

	from util import pickle_fits
	pickle_fits(writefilename+'.fits', result)	



def __fit_map(
	map_as_dict,
	keys,
	model,
	nfits,
	maxitr,
	ncpu,
	debug=False,
	debug_output_path='../'
	):
	if model=='dog': 	 from dog  import bestfit
	elif model=='edog':	 from edog import bestfit

	def func(chunk, patch_w, channel_w, mode):
		key = chunk['n']
		rf = chunk['rf']
		p, err = bestfit(mode, 
						 rf, 
						 channel_w, 
						 patch_w, 
						 key=key, 
						 num_attempts=nfits, 
						 maxiter=maxitr, 
						 debug=debug, 
						 return_error=True,
						 debug_output_path=debug_output_path)
		return {'n':key,'p':p,'err':err}	

	import numpy as N
	from ..base.receptivefield import is_close_to_zero
	data = []
	for key in keys:
		rf = map_as_dict['W'][key]
		abs_rf = N.abs(rf)
		abs_max = N.max( abs_rf )
		abs_min = N.min( abs_rf )
		value_spectrum = abs_max - abs_min
		if value_spectrum > 0.2 and \
		 not is_close_to_zero(rf, verbose=False, atol=1e-02):
			data.append({'n':key, 'rf':rf})

	from util import multiproc_fit2
	fits = multiproc_fit2(
		nprocs=ncpu, 
		verbose=True, 
		data=data, 
		func=func, 
		args=(map_as_dict['vis'], map_as_dict['vis']**2, map_as_dict['mode']))
	return fits






def print_pickled_fits(
	map_file,
	complete=False,
	):
	from util import depickle_fits
	dic = depickle_fits(map_file, exit_on_errer=False)
	if not dic:
		dic = depickle_fits(map_file, suffix='cfits')

	from pprint import pprint
	if not complete:
		keys = dic.keys()
		for key in keys:
			print key
			if type(dic[key]) == dict:
				if key=='map' or key=='dic_cluster':
					for k in dic[key].keys():
						print '\t', k, ':', dic[key][k]
				else:
					print '\t', dic[key].keys()
			else:
				print '\t', dic[key]
	else:	
		print pprint(dic)





def rec_map_from_pickled_result(
	map_file,
	odir
	):
	from util import depickle_fits
	dic = depickle_fits(map_file)

	W_rec = __rec_map(dic['map'], dic['fits'], dic['model'])

	from ..base import make_working_dir_sub
	work_dir = make_working_dir_sub(odir, 'rec')

	from ..base import make_filename
	writefilename = make_filename(map_file,
		'rec_'+dic['model']+'_rgc',
		'.png', odir=work_dir)

	from ..base.plots import save_w_as_image
	save_w_as_image(W_rec, dic['map']['vis'], dic['map']['vis'], dic['map']['hid'], dic['map']['hid'], 
		mode=dic['map']['mode'], 
		outfile=writefilename, 
		verbose=False)

	print 
	print 'reconstruction ', writefilename+'.png', ' written.'



def __rec_map(
	map_as_dict,
	fits,
	model
	):
	import numpy as N
	if model=='dog': 	from dog  import reconstruct
	else:			 	from edog import reconstruct

	rec_W = N.zeros(map_as_dict['W'].shape)
	for key in fits:
		fit = fits[key]
		p = fit['p']
		rec_W[key] = reconstruct(p, 
								 map_as_dict['mode'], 
								 map_as_dict['vis']**2, 
								 map_as_dict['vis'], 
								 map_as_dict['W'][key].shape)
	return N.copy(rec_W)




