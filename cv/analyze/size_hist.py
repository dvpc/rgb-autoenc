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

import matplotlib.pyplot as plt


def make_size_hist(
	args_file,
	args_odir,
	odir_nosubdir = False,
	luminosity_channels = False,
	):
	import numpy as N


	plt.rcParams['xtick.major.pad']='12'
	plt.rcParams['ytick.major.pad']='12'
	plt.rc('axes', linewidth = 0)


	'''loading precomputed fits'''
	from util import depickle_fits
	res = depickle_fits(args_file, suffix='cfits')
	

	'''get fit and cluster data'''
	fits = res['fits']
	fit_keys = fits.keys()
	cl = res['dic_cluster']
	num_channel = cl['num_clusters']
	colors = cl['prototype_cluster']

	perc_cluster = N.round(N.array(cl['num_each_cluster'])/len(fit_keys)*100)


	model = res['model']
	assert model=='edog'

	'''y axis: mean size of RF'''
	'''x axis: anzahl der RF'''

	areas_by_cluster = [[] for c in [None]*num_channel]

	keys = sorted(fit_keys)
	for i, key in enumerate(keys):
		clid = fits[key]['cl']

		'''params: 0: mu_x  1: mu_y  2: sigma_x  3: sigma_y  4: c_to_s_ratio
			   5: theta (rotation) 6: k_s (k_c is implicitly fixed as 1)
			   7: cdir_a, 8: cdir_b,  9: cdir_c
			   10: bias_a, 11: bias_b, 12: bias_c
		'''		
		p = fits[key]['p']
		sigma = p[2:4]
		c_to_s_ratio = p[4]
		''''RF size'''
		c_area = N.pi * sigma[0] * sigma[1] 
		s_area = c_area * c_to_s_ratio
		# apprarea = N.max([c_area, s_area])
		apprarea = N.mean([c_area, s_area])
		
		arr_area = areas_by_cluster[clid]
		arr_area.append(apprarea) 


	data = []
	for i, a in enumerate(areas_by_cluster):
		data.append((i, cl['num_each_cluster'][i], N.mean(a), int(perc_cluster[i])))
	sdata = sorted(data, key=lambda x: x[1])

	fig = plt.figure()
	ax = fig.add_subplot(111)


	perc_sum = 0
	for i, a in enumerate(sdata):
		ax.bar(perc_sum, a[2], a[3], color=colors[a[0]], alpha=.9, linewidth=0)
		perc_sum += a[3]
	ax.get_yaxis().set_ticks(N.arange(0,9,2))
	ax.get_xaxis().set_ticks(N.arange(0,101,25))


	# labels = []
	# for i, a in enumerate(sdata):
	# 	ax.bar([i], [a[2]], 1, color=colors[a[0]], alpha=1., linewidth=0, align='center')
	# 	labels.append(int(a[1]))
	# 	for m in [ax.title] + ax.get_xticklabels() + ax.get_yticklabels(): 
	# 		m.set_fontsize(18)
	# ax.get_xaxis().set_ticklabels(labels)
	# ax.get_xaxis().set_ticks(N.arange(0,8,1))
	# ax.get_yaxis().set_ticks(N.arange(0,8,1))
	# # ax.get_yaxis().set_ticks([])
	# ax.set_aspect(1)
	# # ax.set_aspect(float(7./8))


	from ..base import make_filename	
	fname = make_filename(args_file,'hist_RF_sizes','.png', './')

	plt.savefig(fname, bbox_inches='tight', dpi=144)
	plt.close(fig)









	exit()



	if num_channel == 3:	labels = list('rgb')
	if num_channel == 5:	labels = list('wbrgb')
	if num_channel == 6:	labels = list('wbrgcm') if luminosity_channels else list('rgbcmy')



	data = N.zeros((N.max(cl['num_each_cluster']), num_channel))
	print data.shape
	for i in xrange(0, data.shape[0]):
		for j in xrange(0, data.shape[1]):
			if i < len(areas_by_cluster[j]):
				data[i,j] = areas_by_cluster[j][i]
	

	fig = plt.figure()
	ax = fig.add_subplot(111)





	# data = N.random.lognormal(size=(37, 6), mean=1.5, sigma=1.75)
	# labels = list('rgbcmy')
	ax.boxplot(data, labels=labels)

	# ax.set_yscale('log')
	# ax.set_yticklabels([])


	from ..base import make_filename	
	fname = make_filename(args_file,'hist_RF_sizes','.png', './')

	plt.savefig(fname, bbox_inches='tight', dpi=288)
	plt.close(fig)




	exit()

	


	assert num_channel == 3 or num_channel == 5 or num_channel == 6
	if num_channel == 3:	rows, cols = 1, 3
	if num_channel == 5:	rows, cols = 2, 3
	if num_channel == 6:	rows, cols = 2, 3
	fig, subplots = plt.subplots(nrows=rows, ncols=cols, sharex=False, sharey=False, squeeze=False)

	for r in range(0, rows):
		for c in range(0, cols):
			ax = subplots[r,c]
			ax.set_aspect(1)
			for m in [ax.title] + ax.get_xticklabels() + ax.get_yticklabels(): m.set_fontsize(10)

			i = r*cols+c
			if i >= num_channel: break
			areas = areas_by_cluster[i]

			areas_max = len(keys)/num_channel/2
			index = N.arange(0, areas_max+1, int(areas_max/4))
			print len(index), index

			X, Y = N.histogram(areas, bins=len(index))
			print X, Y
			# ax.bar(index, Y[0:len(index)], 1, color=colors[i], alpha=1.)





			# ax.hist(areas, bins=10, normed=False, color=colors[i], alpha=1., histtype='stepfilled')
			# num_max = len(keys)/num_channel/1.3  #len(areas)/2
			# ax.get_yaxis().set_ticks(N.arange(0, num_max+1, int(num_max/4)))
			
			# size_max = 20 #N.max(areas)			
			# ax.get_xaxis().set_ticks(N.arange(0, size_max+1, int(N.ceil(size_max/5))))
			# ax.set_aspect(float(size_max/num_max))
			# for m in [ax.title] + ax.get_xticklabels() + ax.get_yticklabels(): m.set_fontsize(8.5)




	exit()

	if not odir_nosubdir:
		import os
		mod_odir = os.path.join(args_odir, '../')
		from ..base import make_working_dir_sub
		work_dir = make_working_dir_sub(mod_odir, 'hist')
	else:
		work_dir = args_odir

	from ..base import make_filename
	fname = make_filename(args_file,'hist_RF_sizes','.png', work_dir)

	fig.subplots_adjust(hspace=0.25, wspace=0.25)
	plt.savefig(fname, bbox_inches='tight', dpi=288)
	plt.close(fig)














