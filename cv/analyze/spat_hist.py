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

def __filter_loc(data, loc):
	filtered = []
	for i in xrange(0,len(data)):
		if data[i][0][0] == loc[0] and data[i][0][1] == loc[1]:
			filtered.append(data[i])
	return filtered	

def __filter_loc_fromto(data, loc_from, loc_to):
	filtered = []
	for i in xrange(0,len(data)):
		if data[i][0][0] >= loc_from[0] and data[i][0][1] >= loc_from[1] and\
		   data[i][0][0] <= loc_to[0] and data[i][0][1] <= loc_to[1]:
			filtered.append(data[i])
	return filtered	


def __filter_cl(data, cl):
	filtered = []
	for i in xrange(0,len(data)):
		if data[i][1]['cl'] == cl:
			filtered.append(data[i])
	return filtered	

def __filter_ch(data, ch):
	filtered = []
	for i in xrange(0,len(data)):
		if data[i][1]['ch'] == ch:
			filtered.append(data[i])
	return filtered	

def __filter_value(data, value, band=1e-02):
	filtered = []
	for i in xrange(0,len(data)):
		if data[i][1]['value'] >= value-band and data[i][1]['value'] <= value+band:
			filtered.append(data[i])
	return filtered	



def make_spatal_hist(
	args_file,
	args_odir,
	abs_weight_threshold=0.7,
	):

	'''loading precomputed fits'''
	from util import depickle_fits
	res = depickle_fits(args_file, suffix='cfits')

	'''retrieving weightmatrix from fits dic'''
	map_as_dict = res['map']
	W = map_as_dict['W']
	vis = map_as_dict['vis']
	ch_width = vis**2
	
	mode = map_as_dict['mode']
	assert mode == 'rgb'

	'''get fit and cluster data'''
	fits = res['fits']
	fit_keys = fits.keys()
	cl = res['dic_cluster']
	num_cluster = cl['num_clusters']

	spat_hist = []

	def make_record(loc, key, cl, ch, value):
		if abs(value) > abs_weight_threshold:
			return (loc, {'key':key, 'cl':cl, 'ch':ch, 'value':value})
		else:
			return None

	keys = sorted(fit_keys)
	for i, key in enumerate(keys):
		cl = fits[key]['cl']
		rf = W[key]
		for j in xrange(0,ch_width):
			sploc = (j/vis,j%vis)

			vr = make_record(loc=sploc, key=key, cl=cl, ch=0, value=rf[j])
			if vr: spat_hist.append(vr)
			vg = make_record(loc=sploc, key=key, cl=cl, ch=1, value=rf[ch_width+j])
			if vg: spat_hist.append(vg)
			vb = make_record(loc=sploc, key=key, cl=cl, ch=2, value=rf[2*ch_width+j])
			if vb: spat_hist.append(vb)

	# import pprint as pp
	# # print pp.pprint(spat_hist)


	# # print pp.pprint( __filter_ch(spat_hist, 0) )
	# # print pp.pprint( __filter_value(spat_hist, 0.2) )
	# print pp.pprint(__filter_cl(
	# 					__filter_ch( 
	# 						__filter_loc_fromto(spat_hist, (1,1), (3,3)),
	# 					1),
	# 				0) )
	

	def hist_2d(ch, cl, onoff=None):
		'''make a 2d histogramm'''
		import numpy as N
		hist_2d = N.zeros((vis,vis), dtype=float)
		def sign_ok(on, val):
			if on==None: return True
			else:
				if val > 0 and on: return True
				elif val < 0 and not on: return True
				else: return False
		for i in xrange(0,len(spat_hist)):
			loc = spat_hist[i][0]
			if spat_hist[i][1]['ch'] == ch and \
			   spat_hist[i][1]['cl'] == cl and \
			   sign_ok(onoff, spat_hist[i][1]['value']): 
				# hist_2d[loc[0],loc[1]] += abs(spat_hist[i][1]['value']) #1
				hist_2d[loc[0],loc[1]] += 1
		return hist_2d		



	import os
	mod_odir = os.path.join(args_odir, '../')
	from ..base import make_working_dir_sub
	work_dir = make_working_dir_sub(mod_odir, 'hist')
	from ..base import make_filename



	def make_plots_of_channel(ch, ison=None, ch_name=None):
		plots = []
		str_name = ch_name+' ' if ch_name != None else ''
		for cl in xrange(0,num_cluster):
			plots += [{'name':str_name+' cluster: '+str(cl), 
					   'value':hist_2d(ch, cl, ison), 
					   'maxmin':False, 
					   'cmap':'Greys', 
					   'balance':True, 
					   'invert':True, 
					   'patch_width':vis, 
					   'interp':'nearest'}]
					   #'catrom'}]
		return plots		


	def hist_input(ch, ison=None, ch_name=None):
		plots = make_plots_of_channel(ch, ison, ch_name)

		if ison==None:		miscstr = ''
		elif ison:			miscstr = '_ON_'
		else:				miscstr = '_OFF_'
		fname = make_filename(args_file, 
			'thresh'+str(abs_weight_threshold)+'_hist_inputch_'+str(ch)+miscstr,'.png', work_dir)

		from ..base.plots import write_row_col_fig
		write_row_col_fig(plots, rows=2, cols=3, filepath=fname+'.png', dpi=144, alpha=1.0, fontsize=5.5)
		print 'file:', fname+'.png', 'written.'


	# hist_input(2, ch_name='Blue')
	# hist_input(1, ch_name='Green')
	# hist_input(0, ch_name='Red')

	hist_input(2, ison=True, ch_name='input Blue ON')
	hist_input(2, ison=False, ch_name='input Blue OFF')
	hist_input(1, ison=True, ch_name='input Green ON')
	hist_input(1, ison=False, ch_name='input Green OFF')	
	hist_input(0, ison=True, ch_name='input Red ON')
	hist_input(0, ison=False, ch_name='input Red OFF')



	def hist_complete(onoff=None):
		plots = []

		for inch in xrange(0,3):
			plots += make_plots_of_channel(inch)
			if onoff!=None:
				plots += make_plots_of_channel(inch, onoff)
				plots += make_plots_of_channel(inch, not onoff)

		fname = make_filename(args_file, 
			'thresh'+str(abs_weight_threshold)+'_hist_','.png', work_dir)

		from ..base.plots import numplots_to_rowscols
		rows, cols = numplots_to_rowscols(num_cluster*9)

		print 'plen', len(plots)
		print rows, cols

		from ..base.plots import write_row_col_fig
		write_row_col_fig(plots, rows=rows, cols=cols, filepath=fname+'.png', dpi=144, alpha=1.0, fontsize=12.5)
		print 'file:', fname+'.png', 'written.'

	# hist_complete(True)





