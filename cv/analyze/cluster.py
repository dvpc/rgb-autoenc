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
		depickled_fits=res, maxcolors=args.maxc, 
		cntrpar=args.cpar, srndpar=args.spar, 
		rfspread=args.sprd, nopix=args.nopix)

	'''cluster obs'''
	idx = __apply_cluster_alg(cluster_data=cluster_data, alg=args.alg,
		prior_cluster_num=args.n, whiten=args.white)

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

	import colorsys, copy
	s_prototype_color = copy.copy(prototype_color)
	s_prototype_color.sort(key=lambda rgb: colorsys.rgb_to_hsv(*rgb))
	print pp(N.round(s_prototype_color,1))



	sorted_ids = []
	sort_ch = [None]*args.n
	for i, pr in enumerate(prototype_color):
		sid = -1
		for j, spr in enumerate(s_prototype_color):
			if pr[0] == spr[0] and pr[1] == spr[1] and pr[2] == spr[2]:
				sid = j
				break
		sorted_ids.append( sid )
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

	return fname+'.cfits'


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


	cluster_dir = __handle_outputdir(args.odir, args.odirnosub)

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


	cluster_dir = __handle_outputdir(args.odir, args.odirnosub)


	if args.plain: plain_str = 'plain'
	else: 		   plain_str = ''

	fname = __extract_name_of_filepath(args.file) + '__' + plain_str
	from ..base import make_filename
	filepath = make_filename(args.file, fname, '.png', odir=cluster_dir)

	__plot_clusters_coverage(
		filepath=filepath,
		res=res,
		args_plain=args.plain,
		args_scale=args.scale,
		args_a=args.a,
		args_pad=args.pad,
		args_dpi=args.dpi,
		args_nosd=args.nosd
	)


def plot_ellipsoid_areas(
	args
	):
	'''loading precomputed fits'''
	from util import depickle_fits
	res = depickle_fits(args.file, suffix='cfits')

	plot_data = []
	for key in res['fits'].keys():
		fit = res['fits'][key]
		p = fit['p']

		if res['model']=='dog':
			c_area = N.pi * p[2] * p[3]
			s_area = c_area * p[4] # c_to_s_ratio
		elif res['model']=='edog':
			c_area = N.pi * p[2] * p[3]
			s_area = c_area * p[4] # c_to_s_ratio
		elif res['model']=='edog_ext':	
			c_area = N.pi * p[2] * p[3] 
			s_area = N.pi * p[8] * p[9] 
		'''center, surround area and cluster index'''
		plot_data.append([c_area, s_area, fit['cl']])

	
	data_arr = N.array(plot_data)
	max_cntr, max_srrd = N.max(data_arr.T[0]), N.max(data_arr.T[1])



	markers = res['dic_cluster']['num_clusters']*['.']*3
	markers[0] = '3'
	markers[1] = '8'
	markers[2] = '4'
	markers[3] = 'o'
	markers[4] = '^'
	markers[5] = '>'
	markers[6] = 'd'
	markers[7] = 'x'
	markers[8] = 'h'
	markers[9] = 'H'

	import matplotlib.pyplot as plt	
	fig = plt.figure()
	import matplotlib.gridspec as gridspec
	gs = gridspec.GridSpec(1, 1)
	ax = fig.add_subplot(gs[0,0], projection=None)
	ax.grid = True	
	for k in xrange(0, len(plot_data)):
		cluster_index = plot_data[k][2]
		ax.scatter(plot_data[k][0], plot_data[k][1], 
			c='k', marker=markers[cluster_index], linewidth=.01, s=24., alpha=args.a)


	ax.set_xlabel('center area')
	ax.set_ylabel('surround area')
	ax.set_aspect(float(max_cntr/max_srrd))
	ax.set_xlim([.7,max_cntr])
	ax.set_ylim([.7,max_srrd])
	ax.set_xscale('log')
	ax.set_yscale('log')

	fname = __extract_name_of_filepath(args.file) + '__scatterplot'
	cluster_dir = __handle_outputdir(args.odir, args.odirnosub)
	from ..base import make_filename
	filepath = make_filename(args.file, fname,'.png', odir=cluster_dir)
	plt.savefig(filepath, dpi=args.dpi, pad_inches=args.pad)
	plt.close(fig)
	print 'plotted scatter, file', filepath, 'written.'	



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
	print 'alive hidden units', alive, '(', N.round(float(alive)/float(W.shape[0]),2) ,'% )'
	print '*****+ Map Parameter'
	print pp.pprint([(key, dic_W[key]) if key!='W' else () for key in dic_W.keys()])
	
	print 


	print '*****+ Cluster data'
	print 'num cluster', cl['num_clusters']
	print 'num rf in each cluster', cl['num_each_cluster']
	print 'prototype weights each cluster', N.round(cl['prototype_cluster'],2)
	print '*****+ Cluster Parameters'
	print pp.pprint(cl['args'])



	# TODO cleanup!
	if res['model']=='dog':
		p_names = ['mu_x','mu_y','r_c','r_s',
				'k_s','cdir_a','dir_b','cdir_c','bias_a','bias_b','bias_c']
	elif res['model']=='edog':
		p_names = ['mu_x','mu_y','sigma_x','sigma_y','c_to_s_ratio','theta',
				'k_s','cdir_a','cdir_b','cdir_c','bias_a','bias_b','bias_c']
	elif res['model']=='edog_ext':
		p_names = ['cmu_x','cmu_y',
				'csigma_x','csigma_y','ctheta',
				'ccdir_a','ccdir_b','ccdir_c',
				'ssigma_x','ssigma_y','stheta',
				'scdir_a','scdir_b','scdir_c',
				'bias_a','bias_b','bias_c','k_s']
	elif res['model']=='edog_ext2':
			p_names = ['cmu_x','cmu_y',
				'csigma_x','csigma_y','ctheta',
				'ccdir_a','ccdir_b','ccdir_c',
				'smu_x','smu_y',
				'ssigma_x','ssigma_y','stheta',
				'scdir_a','scdir_b','scdir_c',
				'k_s']
	else:
		p_names = []



	print 
	print '*****+ Fit Parameters'
	for key in res['fits'].keys():
		print '__rf:', key, ':'
		for fitkey in res['fits'][key].keys():
			if fitkey == 'p':
				for idx, par in enumerate(res['fits'][key][fitkey]):
					print '\t', p_names[idx], '  \t', N.round(par,2)
				
			else:
				print fitkey, ':', res['fits'][key][fitkey]
	
		print
	print 



def evaluate_cluster_validity(
	args
	):
	print
	assert args.alg == 'kmean', 'not implemented'

	'''load precomputed fits'''
	from util import depickle_fits
	res = depickle_fits(args.file, suffix='fits')

	'''retrieving weightmatrix from fits dic'''
	W = res['map']['W']
	patch_w = res['map']['vis']
	channel_w = res['map']['vis']**2
	W_mode = res['map']['mode']
	W_clip = res['map']['clip']

	'''generate cluster data'''	
	cluster_data = __collect_cluster_data(W_mode, patch_w, channel_w, W, W_clip, 
		depickled_fits=res, maxcolors=args.maxc, 
		cntrpar=args.cpar, srndpar=args.spar, 
		rfspread=args.sprd, nopix=args.nopix)


	'''prepare result data struc'''
	import sys
	results_num = len(xrange(args.mincl, args.maxcl))
	results = [None]*results_num
	results_davis = [None]*results_num
	for i, c in enumerate(xrange(0,results_num)):
		results[i] = sys.float_info.max
		results_davis[i] = sys.float_info.max



	def __apply_kmean(obs, prior_cl_num):
		if args.white:
			from scipy.cluster.vq import whiten
			obs = whiten(obs)
		from scipy.cluster.vq import kmeans, vq
		centroids,_ = kmeans(obs, prior_cl_num, iter=1250)
		idx, dist = vq(obs,centroids)
		return obs, centroids, idx, dist

	def __compact_data(obs, prior_cl_num, centroids, idx, dist):
		'''put result into explicit data structure'''
		compact = [None]*prior_cl_num
		for i, c in enumerate(centroids):
			compact[i] = []	
		for i, observation in enumerate(obs):
			compact[idx[i]].append([observation, dist[i]])
		return compact



	def __davies_bouldin(prior_cl_num, centroids, dist_by_cluster):
		'''scatter inside a cluster c_i'''
		scatter_arr = [None]*prior_cl_num
		for i, c in enumerate(centroids):
			cluster_len = len(dist_by_cluster[i])
			scatter_inside_cl = 0.0
			for cluster in dist_by_cluster[i]:
				scatter_inside_cl += cluster[-1]
			scatter_arr[i] = scatter_inside_cl/cluster_len		

		'''separation between clusters i and j'''
		separation_max_arr = [None]*prior_cl_num
		for i, c in enumerate(centroids):
			separation_of_cl = []
			for j, c_other in enumerate(centroids):
				if i != j:
					separation_of_cl.append( (scatter_arr[i]+scatter_arr[j])/euclidean(c, c_other) )
			separation_max_arr[i] = N.max(separation_of_cl)

		dbidx = 0
		for i, c in enumerate(centroids):
			dbidx += separation_max_arr[i]
		return dbidx/prior_cl_num


	from scipy.spatial.distance import euclidean
	def __ray_turi(obs, centroids, dist_by_cluster):
		'''intra measure / compactness of clusters -> want MIN'''
		intra = 0
		for i, z in enumerate(centroids):
			for x in dist_by_cluster[i]:
				intra += euclidean(x[0], z)
		intra = intra/len(obs)		

		'''inter cluster measure; dist between cluster centres -> want MAX'''
		inter = sys.float_info.max
		for i, z in enumerate(centroids):
			for j, z_other in enumerate(centroids):
				if i != j and j >= i+1:
					eucl = euclidean(z, z_other)
					if inter > eucl:
						inter = eucl

		validity = intra / inter
		return validity		



	'''iterate through tests'''
	for x in xrange(1,args.ntest):
		sys.stdout.write(str(x))
		sys.stdout.flush()

		'''for each num in range apply clsuter algo and test results'''
		for i, clnum in enumerate(xrange(args.mincl, args.maxcl)):

			'''cluster obs'''
			obs, centroids, idx, dist = __apply_kmean(cluster_data, clnum)
			distances_by_cluster = __compact_data(obs, clnum, centroids, idx, dist)

			sys.stdout.write('.')
			sys.stdout.flush()

			rayturi = __ray_turi(obs, centroids, distances_by_cluster)
			davis = __davies_bouldin(clnum, centroids, distances_by_cluster)

			if results[i] > rayturi:
				results[i] = rayturi
			if results_davis[i] > davis:
				results_davis[i] = davis
			# results[i] += rayturi
			# results_davis[i] += davis


	print 
	print 'map', args.file.split('/')[-1], ',', 
	# print 'map', args_file.split('/')[2], ',', 
	print args.ntest, '''iterations'''
	print '''clusters  \tray turi \tdavies bouldin'''
	for i, c in enumerate(xrange(args.mincl,args.maxcl)):
		print i+3, '\t\t', N.round(results[i],4), '\t\t', N.round(results_davis[i],4)



def __plot_clusters_coverage(
	filepath,
	res,
	args_plain,
	args_scale,
	args_a,
	args_pad,
	args_dpi,
	args_nosd
	):
	args_n = res['dic_cluster']['num_clusters']
	patch_w = res['map']['vis']
	from ..base.plots import numplots_to_rowscols
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
			subplot.annotate(str(fit['n']), fontsize=4, xy=(p[0], p[1]))
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
		elif res['model']=='edog_ext':
			'''params: 0: cmu_x  1: cmu_y  
			   2: csigma_x  3: csigma_y  4: ctheta
			   5: ccdir_a 	6: ccdir_b   7: ccdir_c
			   8: ssigma_x  9: ssigma_y 10: stheta
			  11: scdir_a  12: scdir_b  13: scdir_c
			  14: bias_a   15: bias_b   16: bias_c
			  17: k_s (k_c is implicitly fixed as 1)'''				
			cmux = p[0]; cmuy = p[1]; rwc = p[2]*r_scale; 		rhc = p[3]*r_scale; 	  ctheta = p[4]
			smux = p[0]; smuy = p[1]; rws = p[17]*p[8]*r_scale; rhs = p[17]*p[9]*r_scale; stheta = p[10]
			ellip_c = [cmux, cmuy, rwc, rhc, -180/N.pi*ctheta]
			ellip_s = [smux, smuy, rws, rhs, -180/N.pi*stheta]
		elif res['model']=='edog_ext2':
			'''params: 0: cmu_x  1: cmu_y  
			   2: csigma_x  3: csigma_y  4: ctheta
			   5: ccdir_a 	6: ccdir_b   7: ccdir_c
			   8: smu_x  9: smu_y
			  10: ssigma_x  11: ssigma_y 12: stheta
			  13: scdir_a   14: scdir_b  15: scdir_c
			  16: k_s (k_c is implicitly fixed as 1)
			   '''		
			cmux = p[0]; cmuy = p[1]; rwc = p[2]*r_scale; 	rhc = p[3]*r_scale; ctheta = p[4]
			smux = p[8]; smuy = p[9]; rws = p[10]*r_scale; 	rhs = p[11]*r_scale; stheta = p[12]
			ellip_c = [cmux, cmuy, rwc, rhc, -180/N.pi*ctheta]
			ellip_s = [smux, smuy, rws, rhs, -180/N.pi*stheta]

		c_face_color = fit['color_rgb']
		c_edge_color = 'none'#fit['c_color_rgb']
		s_face_color = fit['s_color_rgb']

		if not args_nosd:
			add_ellipsoid(subplot, par=ellip_s, alpha=args_a*.75, edge_color=s_face_color, face_color='none', linewidth=1., plain=args_plain)
		add_ellipsoid(subplot, par=ellip_c, alpha=args_a, edge_color=c_edge_color, face_color=c_face_color, linewidth=.5, plain=args_plain)


	plt.tight_layout(pad=0, w_pad=args_pad, h_pad=args_pad)
	plt.savefig(filepath+'.png', dpi=args_dpi, bbox_inches='tight', pad_inches=.1)
	plt.close(fig)
	print 'plotted coverage', filepath+'.png', 'written.'




 
def __collect_cluster_data(
	W_mode,
	patch_w,
	channel_w,
	W,
	W_clip,
	depickled_fits={},
	maxcolors=False,
	cntrpar=False, 
	srndpar=False,
	rfspread=.0,
	nopix=False
	):
	from parametric_fit import __rec_map
	W_rec = __rec_map(depickled_fits['map'], depickled_fits['fits'], depickled_fits['model'])

	'''generate cluster data'''
	import numpy as N

	cluster_data = []

	from ..base.plots import normalize_color
	from ..base.receptivefield import value_of_rfvector_at

	keys = sorted(depickled_fits['fits'].keys())
	for key in keys:
		fit = depickled_fits['fits'][key]
		p, n = fit['p'], fit['n']


		'''store coords of position with maximal variance
		for later usage of chosing prototypical filters'''
		real_pix_center_coords = ( N.round(p[0]) , N.round(p[1]) )
		fit['real_pix_center_coords'] = real_pix_center_coords


		'''center color the traditional way'''
		color = value_of_rfvector_at(W_mode, W[n], real_pix_center_coords[0], real_pix_center_coords[1], patch_w, channel_w)
		color = N.array([color[0], color[1], color[2]])
		cntr_color_dir = N.array([p[5], p[6], p[7]])
		srnd_color_dir = N.array([p[11], p[12], p[13]])

		if maxcolors:			
			cntr = __transpose_zero_to_one( normalize_color(cntr_color_dir) )
			srrnd = __transpose_zero_to_one( normalize_color(srnd_color_dir) )
			color = __transpose_zero_to_one( normalize_color(color) )

		else:
			cntr = __transpose_zero_to_one( cntr_color_dir )
			srrnd = __transpose_zero_to_one( srnd_color_dir )
			color = __transpose_zero_to_one( color )


		fit['color_rgb'] = color
		if not nopix:
			arr = N.copy(color)
		else:
			arr = []

		fit['c_color_rgb'] = cntr
		fit['s_color_rgb'] = srrnd

		if cntrpar:
			arr = N.append(arr, cntr_color_dir)
		if srndpar:
			arr = N.append(arr, srnd_color_dir)
		if rfspread > 0:
			'''mean of spread'''
			spm = (p[2] + p[3])/rfspread
			arr = N.append(arr, spm)

		cluster_data.append(arr)

	return cluster_data


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





def __apply_cluster_alg(
	cluster_data=[],
	alg='kmean',
	prior_cluster_num=2,
	whiten=False
	):
	pass
	'''clustering'''
	if alg == 'kmean':
		if whiten:
			from scipy.cluster.vq import whiten
			cluster_data = whiten(cluster_data)
		from scipy.cluster.vq import kmeans, vq
		centroids,_ = kmeans(cluster_data, prior_cluster_num, iter=1250)
		idx, dist = vq(cluster_data,centroids)
		return idx
	elif alg == 'spec':
		X = cluster_data
		if whiten:
			from sklearn.preprocessing import StandardScaler
			X = StandardScaler().fit_transform(X)
		from sklearn import cluster
		spectral = cluster.SpectralClustering(n_clusters=prior_cluster_num, eigen_solver='arpack')
		spectral.fit(X)
		import numpy as N
		idx = spectral.labels_.astype(N.int)
		return idx
	else:
		assert False, "Not implemented"


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
						   'n':args.n, 
						   'alg':args.alg, 
						   'fold':args.fold}
		args_file = args.file
	else:
		cl_dict['num_clusters'] = args['n']
		cl_dict['args'] = args
		args_file = args['file']

	depickled_fits['dic_cluster'] = cl_dict

	fname = __extract_name_of_filepath(args.file) + '__' + __build_filename_from_args(args)
	
	from ..base import make_filename
	writefilename = make_filename(args_file, fname, '.cfits', odir=odir)

	from util import pickle_fits
	pickle_fits(writefilename+'.cfits', depickled_fits)
	return writefilename




def __extract_name_of_filepath(fpath):
	import os
	return os.path.split(fpath)[-1].split('.')[0]


def __handle_outputdir(odir, odirnosub=False):
	if odir.find('cl/') > -1:
		return odir
	else:
		if not odirnosub:
			import os
			mod_odir = os.path.join(odir, '../')		
			from ..base import make_working_dir_sub		
			cluster_dir = make_working_dir_sub(mod_odir, 'cl')
		else:
			import os
			cluster_dir = odir
			cluster_dir = os.path.join(cluster_dir, 'cl/')		
		return cluster_dir


def __build_filename_from_args(
	args
	):
	'''compute filename'''
	import argparse
	str_fold = ''
	if type(args) == argparse.Namespace:
		if args.fold: str_fold = 'fold_'
		str_n = str(args.n)
		str_alg = str(args.alg)
		str_sprd = '_sprd_' + str(int(args.sprd)) if args.sprd != -1 else ''
		str_white = '_white_' + str(args.white) if args.white else ''
	else:
		if args['fold']: str_fold = 'fold_'
		str_n = str(args['n'])
		str_alg = str(args['alg'])
		str_sprd = '_sprd_' + str(int(args['sprd'])) if args['sprd'] != -1 else ''
		str_white = '_white_' + str(args['white']) if args['white'] else ''

	str_thr = str_n + 'ch' + str_sprd + str_white
	misc = str_alg + '_' + str_fold + str_thr 
	return misc



import numpy as N
def __transpose_zero_to_one(
	a
	):
	'''transpose color from [-1, 1] to [0, 1]'''
	# return (a + 1.)/2.
	return N.clip((a + 1.)/2., 0, 1)


