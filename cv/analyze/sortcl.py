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


def sort_cl_fits(
	args_file,
	args_odir,
	indices=None,
	):

	'''loading precomputed fits'''
	from util import depickle_fits
	res = depickle_fits(args_file, suffix='cfits')

	'''get fit and cluster data'''
	cl = res['dic_cluster']
	num_cluster = cl['num_clusters']
	num_indices = len(indices)

	assert num_cluster <= num_indices, '\ninsufficient indices given: '+str(num_indices)+' num clusters: '+str(num_cluster)

	import numpy as N
	new_order = N.arange(num_cluster)
	for i, index in enumerate(indices):
		if i == num_cluster:
			break
		new_order[i] = index

	idx = cl['cluster_index_list']
	num_cl = cl['num_each_cluster']
	prot_cl = cl['prototype_cluster']

	s_num_cl = [num_cl[i] for i in new_order]
	s_idx = N.copy(idx)
	s_prot_cl = [prot_cl[i] for i in new_order] 
	for i in xrange(0,len(new_order)):
		s_idx[idx==new_order[i]] = i
		s_num_cl[i] = num_cl[new_order[i]]

	keys = sorted(res['fits'].keys())
	for i, key in enumerate(keys):
		res['fits'][key]['cl'] = s_idx[i]


	import os
	mod_odir = os.path.join(args_odir, '../')
	from ..base import make_working_dir_sub
	cluster_dir = make_working_dir_sub(mod_odir, 'cl')

	'''write clustered data'''
	cl['args']['file'] = args_file
	from cluster import __write_clustered_fits
	fname = __write_clustered_fits(cl['args'], 
		num_types_list=s_num_cl,
		prototype_color=s_prot_cl, 
		idx_list=s_idx,
		depickled_fits=res,
		odir=cluster_dir,
		num_dead_rf=cl['num_dead_rf'],
		per_dead_rf=cl['per_dead_rf'])


	print fname+'.cfits written'





