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


def cluster_map(
	args
	):
	import numpy as N

	'''load precomputed fits'''
	from util import depickle_fits
	res = depickle_fits(args.file)


	'''retrieving weightmatrix from fits dic'''
	W = res['map']['W']
	patch_w = res['map']['vis']
	channel_w = res['map']['vis']**2
	W_mode = res['map']['mode']
	W_clip = res['map']['clip']



	rf_total = W.shape[0]
	rf_non_zeros = len(res['fits'].keys())
	rf_zeros = rf_total - rf_non_zeros
	rf_dead_perc = N.round(rf_zeros/float(rf_total),2)
	print
	print rf_zeros, 'of', rf_total, 'RF are zeros.' 
	print 'ratio dead/total:', rf_dead_perc
	print



	'''generate cluster data'''
	cluster_data = __collect_cluster_data(W_mode, patch_w, channel_w, W, W_clip, 
		depickled_fits=res, reconstr=args.rec, csp=args.csp, chrm=args.chr, 
		err=args.err, nz=args.nz, surround=args.surr)



	'''cluster obs'''
	idx, args.n = __apply_cluster_alg(cluster_data=cluster_data, alg=args.alg,
		prior_cluster_num=args.n, t=args.t)


	'''assign cluster id to fits'''
	num_types = N.zeros(args.n)
	proto_color_arr = [[] for prot in [None]*args.n]
	keys = sorted(res['fits'].keys())
	for i, key in enumerate(keys):
		fit = res['fits'][key]
		num_types[idx[i]] += 1
		proto_color_arr[idx[i]].append(fit['color_rgb'])
	prototype_color = [N.mean(prot, axis=0) for prot in proto_color_arr]



	from pprint import pprint as pp
	# print 'proto colors'
	# print pp(N.round(prototype_color,1))

	'''sort prototype colors ON / OFF super ugly'''
	val_proto_on, val_proto_off = [], []
	for pr in prototype_color:
		fmax = N.max(pr)
		on = fmax > .7 # can fail on not enough converged maps
		if on:	val_proto_on.append( (pr[0], pr[1], pr[2], N.argmax(pr)) )
		else:	val_proto_off.append( (pr[0], pr[1], pr[2], N.argmin(pr)) )

	p_dtype = [('r', float), ('g', float), ('b', float), ('id', int)]
	proto_on = N.sort(N.array(val_proto_on, dtype=p_dtype), order=['id'])
	proto_on_flat = [[pr[0],pr[1],pr[2]] for pr in proto_on]
	proto_off = N.sort(N.array(val_proto_off, dtype=p_dtype), order=['id'])
	proto_off_flat = [[pr[0],pr[1],pr[2]] for pr in proto_off]

	if proto_on_flat == []:
		s_prototype_color = proto_off_flat
	elif proto_off_flat == []:
		s_prototype_color = proto_on_flat
	else:
		s_prototype_color = N.concatenate([
			[[pr[0],pr[1],pr[2]] for pr in proto_on],
			[[pr[0],pr[1],pr[2]] for pr in proto_off]
		])

	sorted_ids = []
	sort_ch = [None]*args.n
	for i, pr in enumerate(prototype_color):
		sorted_ids.append( N.where(s_prototype_color==pr)[0][0] )
	for i, sid in enumerate(sorted_ids):
		sort_ch[sid] = i

	s_num_types = [num_types[i] for i in sort_ch]
	s_idx = N.copy(idx)
	for i in xrange(0,len(sort_ch)):
		s_idx[idx==sort_ch[i]] = i
		s_num_types[i] = num_types[sort_ch[i]]


	'''fold opposing channels'''
	if args.fold:
		assert args.n % 2 == 0, 'number of clusters need to be even in order to fold opposing clusters.'
		'''leaving the prototype colors intact - just reducing the s_idx entries by half.
		so magnitude |prototype_color| = 6 is not touched but entries in s_idx originally ranging from 0 to 5
		will be reduced to 0 to 2. so all entries in prototype_color are invalid in terms of real clusters contents.
		... ugly'''
		fold_n = args.n/2
		fold_num_types = [0 for i in xrange(0,args.n)]
		fold_idx = N.copy(s_idx)
		from math import ceil
		for i in xrange(0,fold_n):
			fold_i = int(ceil(fold_n + i))
			# fold_idx[fold_idx==i] = i
			fold_idx[fold_idx==fold_i] = i
			fold_num_types[i] = s_num_types[i] + s_num_types[fold_i]


		s_idx = N.copy(fold_idx)
		s_num_types = N.copy(fold_num_types)
		args.n = fold_n

	print 'sorted proto colors'
	print pp(N.round(s_prototype_color,1))
	for i in xrange(0,args.n):
		print i, '\t', int(s_num_types[i]), '\t'	
	print sort_ch
	print s_idx

	'''write cluster membership into fit data'''
	keys = sorted(res['fits'].keys())
	for i, key in enumerate(keys):
		res['fits'][key]['cl'] = s_idx[i]

	from ..base import make_working_dir_sub	
	cluster_dir = make_working_dir_sub(args.odir, 'cl')

	'''write clustered data'''
	fname = __write_clustered_fits(args, 
		num_types_list=N.copy(s_num_types),
		prototype_color=N.copy(s_prototype_color), 
		idx_list=s_idx,
		depickled_fits=res,
		odir=cluster_dir,
		num_dead_rf=rf_zeros,
		per_dead_rf=rf_dead_perc)

	if args.plot:
		from ..base import make_filename
		filepath = make_filename(args.file,'image__' + fname, '.png', odir=cluster_dir)

		__plot_clusters_coverage(
			filepath=filepath,
			res=res,
			args_plain=False,
			args_scale=1.2,
			args_a=1,
			args_pad=.01,
			args_dpi=288
		)		

	if args.pr:
		import os
		from prune import prune_map
		prune_map(args_file=os.path.join(cluster_dir,fname+'.cfits'),args_odir=cluster_dir)




def plot_clusters_colorspace(
	args
	):
	'''loading precomputed fits'''
	from util import depickle_fits
	res = depickle_fits(args.file, suffix='cfits')

	args_n = res['dic_cluster']['num_clusters']
	idx = res['dic_cluster']['cluster_index_list']
	prototype_color = res['dic_cluster']['prototype_cluster']

	cluster_data = []

	keys = sorted(res['fits'].keys())
	for i, key in enumerate(keys):
		fit = res['fits'][key]
		cluster_data.append(fit['color_rgb'])


	'''normalize cluster data'''
	cluster_data -= N.min(cluster_data)
	cluster_data /= N.max(cluster_data)




	'''plot'''
	import matplotlib.pyplot as plt	
	from mpl_toolkits.mplot3d import Axes3D
	Axes3D(plt.figure())
	import matplotlib.gridspec as gridspec

	if args.single:
		gs = gridspec.GridSpec(1, 1)
	 	plt.rcParams['axes.labelsize'] = 18

	else:
		gs = gridspec.GridSpec(3, 1)
		plt.rcParams['axes.linewidth'] = 0.5
	 	plt.rcParams['axes.labelsize'] = 5
	 	plt.rcParams['lines.linewidth'] = .1

	fig = plt.figure()

	markers = args_n*['.']
	markers[0] = 'o'
	markers[1] = 'h'
	markers[2] = 'p'
	markers[3] = '^'
	markers[4] = 'v'
	if args_n > 5: markers[5] = 'd'

	def make_plot(fig, pos, proj='3d', fontsize=9):
		if proj == '3d':
			ax = fig.add_subplot(pos, aspect='equal', projection=proj, xmargin=0)
			ax.set_xlim([.05,.95])
			ax.set_ylim([.05,.95])
			ax.set_zlim([.05,.95])

			ax.set_xlabel('red')
			ax.set_ylabel('green')
			ax.set_zlabel('blue')
			ax.grid(True)
			# ax.xaxis.set_rotate_label(False)
			# ax.yaxis.set_rotate_label(False)
			# ax.zaxis.set_rotate_label(False)
			ax.xaxis._axinfo['tick']['inward_factor'] = 0
			ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
			ax.yaxis._axinfo['tick']['inward_factor'] = 0
			ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
			ax.zaxis._axinfo['tick']['inward_factor'] = 0
			ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
			ax.zaxis._axinfo['tick']['outward_factor'] = 0.4			
			[t.set_va('center') for t in ax.get_yticklabels()]
			[t.set_ha('left') for t in ax.get_yticklabels()]
			[t.set_va('center') for t in ax.get_xticklabels()]
			[t.set_ha('right') for t in ax.get_xticklabels()]
			[t.set_va('center') for t in ax.get_zticklabels()]
			[t.set_ha('left') for t in ax.get_zticklabels()]
			ax.xaxis.pane.fill = False
			ax.yaxis.pane.fill = False
			ax.zaxis.pane.fill = False
			ax.xaxis._axinfo['label']['space_factor'] = 2.8
			ax.yaxis._axinfo['label']['space_factor'] = 2.8
			ax.zaxis._axinfo['label']['space_factor'] = 3.5
			ax.xaxis._axinfo['axisline']['line_width'] = 3.0
			if args.single:
				pass
			else:			
				ax.tick_params(axis='x', labelsize=3.5)
				ax.tick_params(axis='y', labelsize=3.5)
				ax.tick_params(axis='z', labelsize=3.5)

			ax.set_xticks([0.2, 0.5, 0.8])
			ax.set_yticks([0.2, 0.5, 0.8])
			ax.set_zticks([0.2, 0.5, 0.8])

		else:
			ax = fig.add_subplot(pos, projection=proj)
			ax.set_xlim([.0,1])
			ax.set_ylim([.0,1])

			ax.xaxis.labelpad = 2
			ax.yaxis.labelpad = 2
			ax.tick_params(axis='x', labelsize=4, pad=2, length=3, width=.05)
			ax.tick_params(axis='y', labelsize=4, pad=2, length=3, width=.05)
			
			ax.set_xticks([0, 0.2, 0.5, 0.8, 1])
			ax.set_yticks([0, 0.2, 0.5, 0.8, 1])

		return ax

	def make_3d_plot(fig, grid_spec):
		ax = make_plot(fig, grid_spec, proj='3d')
		if args.single:
			s = 84
			lw = .2
		else:
			s = 8
			lw = .1
		# ax.view_init(15, -60)
		for k in xrange(0, args_n):
			ax.scatter(cluster_data[idx==k,0],
					   cluster_data[idx==k,1],
					   cluster_data[idx==k,2], 
					   c=prototype_color[k], marker='.', linewidth=lw, s=s, alpha=1.)

	def make_2d_plot(fig, grid_spec, axis_id, labels=('red', 'green'), fontsize=9):
		ax = make_plot(fig, grid_spec, proj=None)
		if args.plain:
			ax.get_xaxis().set_ticks([])
			ax.get_yaxis().set_ticks([])
			ax.set_title('')
			ax.axis('off')	
		else:		
			ax.set_xlabel(labels[0])
			ax.set_ylabel(labels[1])
		ax.set_aspect(1)
		for k in xrange(0, args_n):
			ax.scatter(cluster_data[idx==k,axis_id[0]],
					   cluster_data[idx==k,axis_id[1]],
					   c=prototype_color[k], marker=markers[k], linewidth=.01, s=6., alpha=1.)
		ax.grid = True
		return ax

	if args.single:
		make_3d_plot(fig, gs[0])		
	else:
		make_2d_plot(fig, gs[0,0], (2,0), ('blue', 'red'), fontsize=12)
		make_2d_plot(fig, gs[1,0], (0,1), ('red', 'green'), fontsize=12)
		make_2d_plot(fig, gs[2,0], (1,2), ('green', 'blue'), fontsize=12)


	if not args.odirnosub:
		import os
		mod_odir = os.path.join(args.odir, '../')		
		from ..base import make_working_dir_sub		
		cluster_dir = make_working_dir_sub(mod_odir, 'cl')
	else:
		cluster_dir = args.odir

	if args.plain: plain_str = 'plain_'
	else: 		   plain_str = ''
	from ..base import make_filename
	filepath = make_filename(
		args.file,'image_clustered_rgc_colorspace_'+plain_str,
		'.png', odir=cluster_dir)

	if args.single:
		plt.savefig(filepath, dpi=args.dpi, pad_inches=args.pad)
	else:
		plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
		plt.savefig(filepath, dpi=args.dpi, pad_inches=args.pad, bbox_inches='tight')

	plt.close(fig)
	print 'plotted cluster data colorspace, file', filepath, 'written.'	



def plot_cluster_cov(
	args
	):
	'''loading precomputed fits'''
	from util import depickle_fits
	res = depickle_fits(args.file, suffix='cfits')

	import os
	mod_odir = os.path.join(args.odir, '../')		
	from ..base import make_working_dir_sub
	cluster_dir = make_working_dir_sub(mod_odir, 'cl')
	if args.plain: plain_str = 'plain_'
	else: 		   plain_str = ''
	from ..base import make_filename
	filepath = make_filename(
		args.file,'image_clustered_rgc_'+plain_str,
		'.png', odir=cluster_dir)

	__plot_clusters_coverage(
		filepath=filepath,
		res=res,
		args_plain=args.plain,
		args_scale=args.scale,
		args_a=args.a,
		args_pad=args.pad,
		args_dpi=args.dpi
	)



def print_cluster_data(
	args
	):

	'''loading precomputed fits'''
	from util import depickle_fits
	res = depickle_fits(args.file, suffix='cfits')


	'''retrieving weightmatrix from fits dic'''
	W = res['map']['W']
	dic_W = res['map']

	'''get fit and cluster data'''
	cl = res['dic_cluster']
	num_dead = cl['num_dead_rf']
	alive = W.shape[0] - num_dead

	import pprint as pp

	print 

	


	print '*****+ Trained map data'
	print 'visible units', W.shape[1]
	print 'hidden units', W.shape[0]
	print 'alive hidden units', alive, '(', N.round(alive/float(W.shape[0]),2) ,'% )'
	print '*****+ Map Parameter'
	print pp.pprint([(key, dic_W[key]) if key!='W' else () for key in dic_W.keys()])
	
	print '*****+ Cluster data'
	print 'num cluster', cl['num_clusters']
	print 'num rf in each cluster', cl['num_each_cluster']
	print 'prototype weights each cluster', N.round(cl['prototype_cluster'],2)
	print '*****+ Cluster Parameters'
	print pp.pprint(cl['args'])





def __plot_clusters_coverage(
	filepath,
	res,
	args_plain,
	args_scale,
	args_a,
	args_pad,
	args_dpi,
	):
	args_n = res['dic_cluster']['num_clusters']
	patch_w = res['map']['vis']
	from ..base.plots import numplots_to_rowscols
	# def numplots_to_rowscols(num):
	# 	sq = int(num**.5)+1
	# 	return sq, sq
	rows, cols = numplots_to_rowscols(args_n)
	import matplotlib.pyplot as plt	
	zeros = N.ones(patch_w**2).reshape(patch_w, patch_w)*.5
	from ..base.plots import add_ellipsoid
	from ..base.plots import prepare_rf_subplot
	fig, subplots = plt.subplots(nrows=rows, ncols=cols, sharex=False, sharey=False, squeeze=False, 
		subplot_kw={'xticks':N.arange(0,patch_w),'yticks':N.arange(0,patch_w),'aspect':1})
	k = 0
	for r in range(0, rows):
		for c in range(0, cols):
			ax = subplots[r,c]
			prepare_rf_subplot(ax, zeros, title=str(k), cmap='Greys', 
				fontsize=4, norm_color=False, display_maxmin=False, plain=args_plain)
			k += 1
	keys = sorted(res['fits'].keys())
	for i, key in enumerate(keys):
		fit = res['fits'][key]
		p = fit['p']
		cl_id = fit['cl']
		subplot = subplots[cl_id/cols, cl_id%rows]
		if not args_plain:
			subplot.annotate(str(fit['n']), fontsize=4, xy=(p[0], patch_w-p[1]))
		r_scale = args_scale
		if res['model']=='dog':
			if p[2] > p[3]: rc = p[2]*r_scale ; rs = p[3]*r_scale
			else: 			rc = p[3]*r_scale ; rs = p[2]*r_scale
			ellip_s = [p[0], patch_w-p[1], rc, rc, 0]
			ellip_c = [p[0], patch_w-p[1], rs, rs, 0]
		elif res['model']=='edog':
			if p[4] > 1: rwc=p[3]*r_scale ; 	 rhc=p[2]*r_scale ; 	 rws=p[3]*p[4]*r_scale ; rhs=p[2]*p[4]*r_scale
			else:		 rwc=p[3]*p[4]*r_scale ; rhc=p[2]*p[4]*r_scale ; rws=p[3]*r_scale ; 	 rhs=p[2]*r_scale
			ellip_c = [p[0], patch_w-p[1], rwc, rhc, 180/N.pi*p[5]]
			ellip_s = [p[0], patch_w-p[1], rws, rhs, 180/N.pi*p[5]]



		color = fit['color_rgb']
		try:
			s_color = fit['s_color_rgb']
		except KeyError:
			s_color = [.1,.1,.1]

		if not args_plain:
			c_face_color = color
			s_face_color = 'none'
		else:
			c_face_color = color
			s_face_color = s_color

		if not args_plain:
			add_ellipsoid(subplot, par=ellip_s, alpha=0.75, edge_color=[.1,.1,.1], face_color=s_face_color, linewidth=.5, plain=args_plain)
		add_ellipsoid(subplot, par=ellip_c, alpha=args_a, edge_color=[.1,.1,.1], face_color=c_face_color, linewidth=.5, plain=args_plain)


	plt.savefig(filepath+'.png', dpi=args_dpi, bbox_inches='tight', pad_inches=args_pad) #
	plt.close(fig)
	print 'plotted coverage', filepath+'.png', 'written.'


def __shape_weight_value(
	mode,
	a,
	):
	import numpy as N	
	if mode=='rgb':
		return N.array(a)
	elif mode=='rg':
		return N.array([a[0], a[1], N.zeros(1)])
	elif mode=='rg_vs_b':
		return N.array([a[0], a[0], a[1]])
	elif mode=='lum':
		return N.array([a[0], a[0], a[0]])

import numpy as N
def __transpose_zero_to_one(
	a
	):
	'''transpose color from [-1, 1] to [0, 1]'''
	# return (a + 1.)/2.
	return N.clip((a + 1.)/2., 0, 1)


def __build_filename_from_args(
	args
	):
	'''compute filename'''
	import argparse
	str_surr = ''
	str_nz = ''
	str_chr = ''
	str_err = ''
	str_fold = ''
	if type(args) == argparse.Namespace:
		if args.surr: str_surr = 'surround_'
		if args.nz:  str_nz = 'nz_'
		if args.chr: str_chr = 'chr_'
		if args.err: str_err = 'err_'
		if args.fold: str_fold = 'fold_'
		str_n = str(args.n)
		str_alg = str(args.alg)
		str_csp = str(args.csp)
		str_t = str(args.t)
	else:
		# if args['surr']: str_surr = 'surround_'
		if args['nz'] != 0: str_nz = 'nz_'
		if args['chr'] != 0: str_chr = 'chr_'
		if args['err'] != 0: str_err = 'err_'
		if args['fold']: str_fold = 'fold_'
		str_n = str(args['n'])
		str_alg = str(args['alg'])
		str_csp = str(args['csp'])
		str_t = str(args['t'])

	if str_alg != 'kmean' and str_alg != 'spec': 	
		str_thr = 't' + str_t + '_' 
	else:											
		str_thr = ''
	misc = str_alg + '_' + str_fold + str_n + 'ch_' + \
		str_thr + str_err + str_nz + str_chr + str_surr + str_csp
	return misc




def __est_real_pix_coords(
	p, 
	rf, 
	W_mode, 
	patch_w, 
	channel_w
	):
	from ..base.receptivefield import value_of_rfvector_at
	'''store coords of position with maximal variance
	for later usage of chosing prototypical filters'''
	real_pix_center_coords = (0,0)
	'''try finding center weight with max variance.
	after 10 attempts of using different perm f the 
	value with max variance is chosen.'''
	perm_f = [N.floor, N.ceil, N.round]
	from ..base import num_channel_of
	val = N.zeros(num_channel_of(W_mode))
	var = 0
	count = 0
	var_max, val_max = 0, N.zeros(val.shape)
	while var < .07 and count < 30:
		perm_f_y = perm_f[N.random.randint(0, len(perm_f))]
		perm_f_x = perm_f[N.random.randint(0, len(perm_f))]
		val = value_of_rfvector_at(W_mode, rf, perm_f_y(p[0]), perm_f_x(p[1]), patch_w, channel_w)
		var = N.var(val) #N.linalg.norm(val)# TODO ????
		if var > var_max:
			var_max = var
			val_max = val
			real_pix_center_coords = ( perm_f_y(p[0]) , perm_f_x(p[1]) )
		count += 1
		# print '+'*count, n, real_pix_center_coords, var
	val = val_max
	# print '-> ', var_max
	return val, real_pix_center_coords



def __est_surround_color(
	p,
	rf,
	W_mode,
	patch_w,
	channel_w
	):
	'''approx. considers only 4 points:
		A: mu + sigma_x,
		B: mu - sigma_x,
		C: mu + sigma_y,
		D: mu - sigma_y'''
	from ..base.plots import rot
	# def rot(angle, xy, p_rot_over):
	# 	s = N.sin(angle)
	# 	c = N.cos(angle)
	# 	xy = (xy[0] - p_rot_over[0], xy[1] - p_rot_over[1])
	# 	rx = xy[0]*c - xy[1]*s
	# 	ry = xy[0]*s + xy[1]*c	
	# 	return (rx+p_rot_over[0], ry+p_rot_over[1])
	mu = (p[0], p[1])
	sigma = (p[2], p[3])
	c_to_s_ratio = p[4]
	theta = p[5]
	sig_x = sigma[0]*c_to_s_ratio
	sig_y = sigma[1]*c_to_s_ratio
	A = rot(-theta, (mu[0]+sig_x, mu[1]), mu)
	B = rot(-theta, (mu[0]-sig_x, mu[1]), mu)
	C = rot(-theta, (mu[0], mu[1]+sig_y), mu)
	D = rot(-theta, (mu[0], mu[1]-sig_y), mu)

	from ..base.receptivefield import value_of_rfvector_at
	# cA = N.array(value_of_rfvector_at(W_mode, rf, N.ceil(A[0]), A[1], patch_w, channel_w))
	# cB = N.array(value_of_rfvector_at(W_mode, rf, N.floor(B[0]), B[1], patch_w, channel_w))
	# cC = N.array(value_of_rfvector_at(W_mode, rf, C[0], N.ceil(C[1]), patch_w, channel_w))
	# cD = N.array(value_of_rfvector_at(W_mode, rf, D[0], N.floor(D[1]), patch_w, channel_w))
	cA = N.array(value_of_rfvector_at(W_mode, rf, A[0], A[1], patch_w, channel_w))
	cB = N.array(value_of_rfvector_at(W_mode, rf, B[0], B[1], patch_w, channel_w))
	cC = N.array(value_of_rfvector_at(W_mode, rf, C[0], C[1], patch_w, channel_w))
	cD = N.array(value_of_rfvector_at(W_mode, rf, D[0], D[1], patch_w, channel_w))

	def absm(v):
		return N.max(N.abs(v))
	def chk_bound(xy):
		lt_zero = xy[0] < 0 or xy[1] < 0
		gt_patch = xy[0] > patch_w or xy[1] > patch_w
		return not (lt_zero and gt_patch)		
	ret, denom = N.zeros(3), 0
	if absm(cA) > .005 and chk_bound(A):	ret += cA ; denom += 1
	if absm(cB) > .005 and chk_bound(B):	ret += cB ; denom += 1
	if absm(cC) > .005 and chk_bound(C):	ret += cC ; denom += 1
	if absm(cD) > .005 and chk_bound(D):	ret += cD ; denom += 1
	if denom == 0:	
		return ret
	else:
		return ret/denom


 
def __collect_cluster_data(
	W_mode,
	patch_w,
	channel_w,
	W,
	W_clip,
	depickled_fits={},
	reconstr=False,
	csp='rgb',
	chrm=0,
	err=0,
	nz=0,
	surround=False,
	func_model='edog',
	):
	'''generate cluster data'''
	import numpy as N
	from ..base.receptivefield import value_of_rfvector_at
	from ..base.receptivefield import chromaticy_of_rfvector
	from ..base.receptivefield import num_relevant_weights
	'''https://docs.python.org/2/library/colorsys.html'''
	import colorsys
	if reconstr:
		if depickled_fits['model'] == 'dog': from fit_dog  import reconstruct
		else: 					  			 from fit_edog import reconstruct

	cluster_data = []

	keys = sorted(depickled_fits['fits'].keys())
	for key in keys:
		fit = depickled_fits['fits'][key]
		p, n = fit['p'], fit['n']
		'''store coords of position with maximal variance
		for later usage of chosing prototypical filters'''
		real_pix_center_coords = (0,0)
		if reconstr:
			rec = reconstruct(p, W_mode, channel_w, patch_w, W[n].shape)
			val = value_of_rfvector_at(W_mode, rec, p[0], p[1], patch_w, channel_w)
			real_pix_center_coords = ( N.round(p[0]) , N.round(p[1]) )
		else:
			val = value_of_rfvector_at(W_mode, W[n], p[0], p[1], patch_w, channel_w)			
			real_pix_center_coords = ( N.round(p[0]) , N.round(p[1]) )
			# val, real_pix_center_coords = __est_real_pix_coords(p, W[n], W_mode, patch_w, channel_w)
		fit['real_pix_center_coords'] = real_pix_center_coords

		c_col = __shape_weight_value(W_mode, val)
		c_col = __transpose_zero_to_one(c_col)
		fit['color_rgb'] = c_col
		fit['color_yiq'] = colorsys.rgb_to_yiq(c_col[0], c_col[1], c_col[2])

		if csp == 'rgb': 
			arr = N.copy(fit['color_rgb'])
		elif csp == 'yiq':  
			arr = N.copy(fit['color_yiq'])
			arr = fit['color_yiq']

		s_col = __est_surround_color(p, W[n], W_mode, patch_w, channel_w)
		s_col = __transpose_zero_to_one(s_col)
		fit['s_color_rgb'] = s_col
		if surround:
			arr = N.append(arr, s_col)

		if err != 0:
			arr = N.append(arr, fit['err']*err)
		if chrm != 0:
			chroma = chromaticy_of_rfvector(W_mode, W[n], patch_w, channel_w)
			arr = N.append(arr, chroma*chrm)
		if nz != 0:
			nonzero = num_relevant_weights(W_mode, W[n], patch_w, channel_w)
			arr = N.append(arr, nonzero*nz)
		cluster_data.append(arr)
	return cluster_data


def __apply_cluster_alg(
	cluster_data=[],
	alg='kmean',
	prior_cluster_num=2,
	t=.155,
	):
	pass
	'''clustering'''
	if alg == 'kmean':
		from scipy.cluster.vq import whiten
		cluster_data = whiten(cluster_data)
		from scipy.cluster.vq import kmeans, vq
		centroids,_ = kmeans(cluster_data, prior_cluster_num, iter=250)
		idx, dist = vq(cluster_data,centroids)
		return idx, prior_cluster_num
	elif alg == 'spec':
		from sklearn import cluster
		from sklearn.preprocessing import StandardScaler
		X = cluster_data
		X = StandardScaler().fit_transform(X)
		spectral = cluster.SpectralClustering(n_clusters=prior_cluster_num, eigen_solver='arpack')
		spectral.fit(X)
		import numpy as N
		idx = spectral.labels_.astype(N.int)
		return idx, prior_cluster_num
	else:
		'''hierarchical clustering
		   http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html'''
		import scipy.cluster.hierarchy as hcluster		   
		'''needs distance matrix: 
		   http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html'''
		import scipy.spatial.distance as dist
		distmat = dist.pdist(cluster_data, 'minkowski')#'euclidean')
		if alg == 'hflat':
			link = hcluster.linkage(distmat)
		elif alg == 'hcomp':
			link = hcluster.complete(distmat)
		elif alg == 'hweight':
			link = hcluster.weighted(distmat)
		elif alg == 'havg':
			link = hcluster.average(distmat)
		idx = hcluster.fcluster(link, t=t, criterion='distance')
		import numpy as N
		post_cluster_num = len(N.unique(idx))
		print '# of channels established:', post_cluster_num
		assert post_cluster_num<64, 'number of cluster too large to be biological meaningful'	
		return idx, post_cluster_num


def __write_clustered_fits(
	args,
	num_types_list=[],
	prototype_color=[],
	idx_list=[],
	depickled_fits={},
	odir=None,
	num_dead_rf=0,
	per_dead_rf=0,
	):
	'''write clustered data'''
	cl_dict =  {'num_each_cluster':num_types_list,
				'prototype_cluster':prototype_color,
				'cluster_index_list':idx_list,
				'num_dead_rf':num_dead_rf,
				'per_dead_rf':per_dead_rf,
			}
	import argparse
	if type(args) == argparse.Namespace:
		cl_dict['num_clusters'] = args.n
		cl_dict['args'] = {'alg':args.alg,
						   'nz':args.nz,
						   'chr':args.chr, 
						   'err':args.err, 
						   'csp':args.csp, 
						   'n':args.n, 
						   't':args.t, 
						   'alg':args.alg, 
						   'rec':args.rec,
						   'fold':args.fold}
		args_file = args.file
	else:
		cl_dict['num_clusters'] = args['n']
		cl_dict['args'] = args
		args_file = args['file']

	depickled_fits['dic_cluster'] = cl_dict

	misc = __build_filename_from_args(args)
	fname = 'clustered_rgc_'+misc
	from ..base import make_filename
	writefilename = make_filename(args_file, fname, '.cfits', odir=odir)

	from util import pickle_fits
	pickle_fits(writefilename+'.cfits', depickled_fits)
	return writefilename





