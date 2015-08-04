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


def make_proto_filter(
	args_file,
	args_odir,
	args_model,
	synthetic_pos=True,
	mean_rec=True,
	max_dist_to_center=2,
	meanlen=None,
	min_err_metric=True,
	abs_weight_threshold=0.7,
	indices=None,
	debug=False,
	odir_nosubdir=False,
	):
	hand_automatic = indices == None

	'''loading precomputed fits'''
	from util import depickle_fits
	res = depickle_fits(args_file, suffix='cfits')

	'''retrieving weightmatrix from fits dic'''
	W = res['map']['W']
	vis = res['map']['vis']
	map_as_dict = res['map']
	assert map_as_dict['mode'] == 'rgb'

	'''get fit and cluster data'''
	fits = res['fits']
	fit_keys = fits.keys()
	model = res['model']

	cl = res['dic_cluster']
	num_channel = cl['num_clusters']

	'''move RF ids in an appropiate data struc, ignore zeros'''
	clusters_by_index = [[] for c in [None]*num_channel]

	keys = sorted(fit_keys)
	for i, key in enumerate(keys):
		cl = fits[key]['cl']
		clusters_by_index[cl].append(key)

	from ..base.receptivefield import convert_rfvector_to_rgbmatrix
	from ..base.plots import colormap_for_mode
	cmap = colormap_for_mode(map_as_dict['mode'])	
	from ..base import make_filename
	from cluster import __transpose_zero_to_one

	if mean_rec:
		'''of every channel mean the fitted parameters'''
		import numpy as N
		num_param = len(fits[fit_keys[-1]]['p'])
		clusters_p_mean = [N.zeros(num_param) for c in [None]*num_channel]
		for c in xrange(0,len(clusters_by_index)):
			cl_ids = clusters_by_index[c]
			if meanlen is None:
				len_cl_id = len(cl_ids)
			else:
				len_cl_id = meanlen
			p_mean_tmp = []
			for i in xrange(0, len_cl_id):
				p = fits[cl_ids[i]]['p']
				p_mean_tmp.append(p)
			p_mean = N.mean(p_mean_tmp, axis=0)
			clusters_p_mean[c] = N.copy(p_mean)
		if model=='dog': from dog  import reconstruct
		else:			 from edog import reconstruct

	else:
		import numpy as N
		hand_by_index = [[] for c in [None]*num_channel]
		zpx = vis/2.
		zpy = vis/2.
		max_absmax = 0

		if not hand_automatic:
			'''use provided indices ... handchosen'''
			for i, index in enumerate(indices):
				if i == num_channel:
					break
				hand_by_index[i] = index
		else:
			'''chose automatically:
			for each channel find the RF with smallest error (and max value > .7), 
			in distance close to the center vis field.'''		
			for c in xrange(0,len(clusters_by_index)):
				cl_ids = clusters_by_index[c]
				len_cl_id = len(cl_ids)
				min_err, min_err_id, min_dist, min_id = N.inf, -1, N.inf, -1
				for i in xrange(0, len_cl_id):
					p = fits[cl_ids[i]]['real_pix_center_coords']
					n = fits[cl_ids[i]]['n']
					err = fits[cl_ids[i]]['err']
					dist = N.sqrt((zpy - p[0])**2 + (zpx - p[1])**2)
					absmax = N.max(N.abs(W[n]))
					if err < min_err and absmax > abs_weight_threshold and dist < max_dist_to_center:
						min_err = err
						min_err_id = n
					if dist < min_dist and absmax > abs_weight_threshold:
						min_dist = dist
						min_id = n
					'''statistic'''
					if absmax > max_absmax:
						max_absmax = absmax
				if min_err_metric:
					hand_by_index[c] = min_err_id
				else:
					hand_by_index[c] = min_id

		'''store distance of fitted center point to center of visual field'''
		trans_rf = []
		for n in hand_by_index:
			if type(n) == list:
				print 'Not enough RF indices for all channels given.\nnum channels:', num_channel, 'num indices:',  len(indices), '\n'
				exit()
			if n == -1:
				print 'No prototype RF found. \nTry lowering wheight treshold\nparameter -thr is:', abs_weight_threshold, 'RF max abs weight:', absmax, '\n'
				exit()
			
			p = fits[n]['real_pix_center_coords']
			dist_y = int(N.floor(zpy - p[0]))
			dist_x = int(N.floor(zpx - p[1]))
			trans_rf.append( (dist_y, dist_x) )
	


	filters = []
	plots = []
	for c in xrange(0,num_channel):
		if mean_rec:
			p = clusters_p_mean[c]
			if synthetic_pos:
				p = N.concatenate([[map_as_dict['vis']/2., map_as_dict['vis']/2.], p[2:]])
			proto_filter = reconstruct(p, 
									 map_as_dict['mode'], 
									 map_as_dict['vis']**2, 
									 map_as_dict['vis'], 
									 map_as_dict['W'][-1].shape)
		else:
			proto_filter = W[hand_by_index[c]]

		'''convert RF vector to matrix and normalize values'''
		proto_filter_matr = convert_rfvector_to_rgbmatrix(
			proto_filter, 
			map_as_dict['vis'],
			map_as_dict['vis'], 
			map_as_dict['mode'])		
		proto_filter_matr = __transpose_zero_to_one(proto_filter_matr)

		'''move RF to visual fields center'''
		if not mean_rec:
			proto_filter_matr = N.roll(proto_filter_matr, trans_rf[c][0], axis=0)
			proto_filter_matr = N.roll(proto_filter_matr, trans_rf[c][1], axis=1)

		plots.append({
			# 'name':('RF: '+str(hand_by_index[c])+' ' if not hand_automatic else '') + 'ch: '+str(c), 
			'name':'RF: '+str(hand_by_index[c])+' ' + ' ch: '+str(c), 
			'value':proto_filter_matr,
			'maxmin':True, 
			'cmap':cmap, 
			'balance':False,
			'invert':False,
			})
		filters.append(N.copy(proto_filter_matr))

	if not mean_rec:
		if hand_automatic:	misc_str = 'auto_'
		else:				misc_str = 'hand_'
		if min_err_metric:	metr_str = 'err_'
		else:				metr_str = 'dist_'
	else:
		misc_str = 'mean_'
		metr_str = ''


	if not odir_nosubdir:
		import os
		mod_odir = os.path.join(args_odir, '../')
		from ..base import make_working_dir_sub
		work_dir = make_working_dir_sub(mod_odir, 'proto')
	else:
		work_dir = args_odir
		
	fname = make_filename(args_file,misc_str+metr_str+'conv_proto','.png', work_dir)
	def numplots_to_rowscols(num):
		sq = int(num**.5)+1
		return sq, sq
	from ..base.plots import write_row_col_fig
	row, col = numplots_to_rowscols(num_channel)
	write_row_col_fig(plots, row, col, fname+'.png', dpi=144, alpha=1.0, fontsize=6)


	dic = {
		'mode': res['map']['mode'],
		'num_chn': num_channel,
		'filters': filters
	}

	writefilename = make_filename(args_file,misc_str+metr_str+'conv_proto','.kern', work_dir)
	from util import pickle_fits
	pickle_fits(writefilename+'.kern', dic)


	if debug:
		from ..base.plots import write_rf_fit_debug_fig
	
		for i, fmat in enumerate(filters):
			fmat = N.swapaxes(fmat, 0, 1)			

			fvec = fmat.reshape(fmat.shape[0]*fmat.shape[1], fmat.shape[2])
			fvecflat = N.copy(N.concatenate([fvec.T[0].T, fvec.T[1].T, fvec.T[2].T]))
			fvecflat -= .5
			fvecflat *= 2.

			p = fits[hand_by_index[i]]['p']
			p[0] = p[0] + trans_rf[i][0]
			p[1] = p[1] + trans_rf[i][1]

			if res['map']['mode'] == 'dog': from dog  import reconstruct
			else: 					  		from edog import reconstruct

			rec = reconstruct(p, res['map']['mode'], vis**2, vis, fvecflat.shape)
			err = (fvecflat - rec)**2
			# err = None

			# fname = make_filename(args_file,str(i)+'_debug','.png', work_dir)
			fname = work_dir + '/' + str(i)+'_debug'+'.png'
			write_rf_fit_debug_fig(fname, fvecflat, vis, 'rgb', p, rec, err, model, 
				scale=2.8, s_scale=3.8, alpha=.5, dpi=300, draw_ellipsoid=True,
				no_title=True, ellipsoid_line_width=1.2)





