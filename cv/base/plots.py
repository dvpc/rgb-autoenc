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
import matplotlib.pyplot as plt


def numplots_to_rowscols(num):
	sq = int(num**.5)+1
	return sq, sq


def rot(angle, xy, p_rot_over):
	s = N.sin(angle)
	c = N.cos(angle)
	xy = (xy[0] - p_rot_over[0], xy[1] - p_rot_over[1])
	rx = xy[0]*c - xy[1]*s
	ry = xy[0]*s + xy[1]*c	
	return (rx+p_rot_over[0], ry+p_rot_over[1])



def colormap_for_mode(
	mode,
	lms=False,
	):
	cmap = None
	if   mode=='lum': 		cmap = 'Greys'
	elif mode=='rg':	 	cmap = 'RdYlGn'
	elif mode=='rg_vs_b': 	cmap = 'RdYlBu'
	return cmap



def normalize_color(
	a
	):
	highS = N.max(a)
	lowS = N.min(a)
	shiftS = lowS
	if shiftS > 0.0: lowS = 0.0
	a = (a - shiftS) / (highS - shiftS)
	return N.copy(a)



from matplotlib.patches import Ellipse
def add_ellipsoid(
	ax, 
	par, 
	alpha=.75, 
	edge_color=[1.0,.0,1.0], 
	face_color='none', 
	linewidth=2.0,
	plain=False):
	e = Ellipse(xy=(par[0], par[1]), width=par[3], height=par[2], angle=par[4], linewidth=linewidth)
	ax.add_artist(e)
	if plain:	e.set_clip_on(False)
	else: 		e.set_clip_box(ax.bbox)
	e.set_edgecolor(edge_color)
	e.set_facecolor(face_color)
	e.set_alpha(alpha)



def prepare_rf_subplot(
	ax,
	a, 
	title='None', 
	fontsize=10, 
	cmap=None,
	display_maxmin=True,
	norm_color=True,
	invert_data=False,
	plain=False
	):
	if not plain:
		if display_maxmin:
			ax.set_title(title + " (" + str(N.round(N.min(a),4)) + "," + str(N.round(N.max(a),4)) + ")")
		else:
			ax.set_title(title)
		for m in [ax.title] + ax.get_xticklabels() + ax.get_yticklabels():
			m.set_fontsize(fontsize)		
		ax.grid()
	else:
		ax.set_title('')
		ax.axis('off')
	if invert_data:
		a *= -1
	if norm_color:
		a = normalize_color(a)
	im = plt.imshow(a, interpolation='nearest', cmap=cmap)
	ax.images.append(im)





import matplotlib.image as pltimg
from ..base.receptivefield import stack_matrix
def save_w_as_image(
	X, 
	in_w, 
	in_h, 
	out_w, 
	out_h, 
	outfile,
	mode,
	dpi=288,
	verbose=True,
	lms=False,
	):
	assert len(X.shape) == 2, 'imagetiles: size error'
	if mode=='rgb': 					vis_ch = 3
	elif mode=='rg' or mode=='rg_vs_b': vis_ch = 2
	else: 								vis_ch = 1
	x = out_h
	y = out_w
	_h = in_h
	_w = in_w
	xy, hw = X.shape
	assert (xy == x*y) and (hw == _h*_w*vis_ch), 'imagetiles: size error'
	if mode=='rgb': 					X = stack_matrix(X, _w, y)
	elif mode=='rg' or mode=='rg_vs_b':	X = stack_matrix(X, _w, y, mode=mode)
	frame = 1	
	if mode=='lum':	Y = N.ones((frame+x*(_h+frame), frame+y*(_w+frame))) * .05
	else:			Y = N.ones((frame+x*(_h+frame), frame+y*(_w+frame), 3)) * .05
	image_id = 0
	for xx in range(x):
		for yy in range(y):
			if image_id >= xy: break
			beginH, beginW = frame+xx*(_h+frame), frame+yy*(_w+frame)
			if mode=='rgb':
				tile = N.array([N.reshape(X[image_id].T[0], (_h, _w)), 
								N.reshape(X[image_id].T[1], (_h, _w)), 
								N.reshape(X[image_id].T[2], (_h, _w))]).T
			elif mode=='rg':
				tile = N.array([N.reshape(X[image_id].T[0], (_h, _w)), 
								N.reshape(X[image_id].T[1], (_h, _w)), 
								N.zeros((_h, _w))]).T
			elif mode=='rg_vs_b':
				tile = N.array([N.reshape(X[image_id].T[0], (_h, _w)), 
								N.reshape(X[image_id].T[0], (_h, _w)), 
								N.reshape(X[image_id].T[1], (_h, _w))]).T
			else:
				tile = N.reshape(X[image_id], (_h, _w))
			Y[beginH : beginH+_h, beginW : beginW+_w] = tile
			image_id += 1
	cmap = colormap_for_mode(mode)
	Y = normalize_color(Y)
	pltimg.imsave(outfile, Y, dpi=dpi, cmap=cmap)
	if (verbose):
		print 'file:', outfile, 'written.'





def write_rf_fit_debug_fig(
	outfile,
	rf,
	patch_width,
	mode,
	p=None,
	rf_reconstr=None,
	rf_reconstr_err=None,
	model='dog',
	scale=2,
	s_scale=None,
	alpha=1.,
	dpi=216,
	draw_ellipsoid=True,
	title=None,
	no_title=False,
	ellipsoid_line_width=.5,
	):
	rf_name = title if title != None else 'RF'
	rec_name = title + 'rec. ' if title != None else 'rec.'

	cmap = colormap_for_mode(mode)
	from receptivefield import convert_rfvector_to_rgbmatrix
	from receptivefield import value_of_rfvector_at
	from ..analyze.cluster import __transpose_zero_to_one
	rf_mat = convert_rfvector_to_rgbmatrix(rf, patch_width, patch_width, mode, swap=False, flip=False)
	rf_mat = __transpose_zero_to_one(rf_mat)
	dic_rf = {'name':'' if no_title else rf_name, 'value':rf_mat, 'maxmin':False, 'cmap':cmap, 'balance':False}
	mu_x, mu_y = 0, 0
	ctheta = 0

	p_names = []
	if not p is None:
		if model=='dog':
			'''     0     1     2        3       ...
			params: mu_x  mu_y  r_c  	 r_s     ...'''
			p_names = ['mu_x','mu_y','r_c','r_s',
				'k_s','cdir_a','dir_b','cdir_c','bias_a','bias_b','bias_c']
			cmux, cmuy = p[0], p[1]
			smux, smuy = p[0], p[1]
			ctheta, stheta = 0, 0
			rhc, rwc = p[2], p[2]
			rhs, rws = p[3], p[3]
		elif model=='edog':
			'''     0     1     2        3        4             5      ...
			params: mu_x  mu_y  sigma_x  sigma_y  c_to_s_ratio  theta  ...'''
			p_names = ['mu_x','mu_y','sigma_x','sigma_y','c_to_s_ratio','theta',
				'k_s','cdir_a','cdir_b','cdir_c','bias_a','bias_b','bias_c']
			cmux, cmuy = p[0], p[1]
			smux, smuy = p[0], p[1]
			ctheta, stheta = p[5], p[5]
			if p[4] > 1: rwc=p[2]*scale ; 	   rhc=p[3]*scale ; 	 rws=p[2]*p[4]*scale ; 	rhs=p[3]*p[4]*scale
			else:		 rwc=p[2]*p[4]*scale ; rhc=p[3]*p[4]*scale ; rws=p[2]*scale ; 	   	rhs=p[3]*scale
		elif model=='edog_ext':
			'''params: 0: cmu_x  1: cmu_y  
			   2: csigma_x  3: csigma_y  4: ctheta
			   5: ccdir_a 	6: ccdir_b   7: ccdir_c
			   8: ssigma_x  9: ssigma_y 10: stheta
			  11: scdir_a  12: scdir_b  13: scdir_c
			  14: bias_a   15: bias_b   16: bias_c
			  17: k_s (k_c is implicitly fixed as 1)'''			
			p_names = ['cmu_x','cmu_y',
				'csigma_x','csigma_y','ctheta',
				'ccdir_a','ccdir_b','ccdir_c',
				'ssigma_x','ssigma_y','stheta',
				'scdir_a','scdir_b','scdir_c',
				'bias_a','bias_b','bias_c','k_s']		
			cmux = p[0]; cmuy = p[1]; rwc = p[2]*scale; 	rhc = p[3]*scale; ctheta = p[4]
			smux = p[0]; smuy = p[1]; rws = p[8]*scale; 	rhs = p[9]*scale; stheta = p[10]
		elif model=='edog_ext2':
			'''params: 0: cmu_x  1: cmu_y  
			   2: csigma_x  3: csigma_y  4: ctheta
			   5: ccdir_a 	6: ccdir_b   7: ccdir_c
			   8: smu_x  9: smu_y
			  10: ssigma_x  11: ssigma_y 12: stheta
			  13: scdir_a   14: scdir_b  15: scdir_c
			  16: k_s (k_c is implicitly fixed as 1)'''		
			p_names = ['cmu_x','cmu_y',
				'csigma_x','csigma_y','ctheta',
				'ccdir_a','ccdir_b','ccdir_c',
				'smu_x','smu_y',
				'ssigma_x','ssigma_y','stheta',
				'scdir_a','scdir_b','scdir_c',
				'k_s']		
			cmux = p[0]; cmuy = p[1]; rwc = p[2]*scale; 	rhc = p[3]*scale; ctheta = p[4]
			smux = p[8]; smuy = p[9]; rws = p[10]*scale; 	rhs = p[11]*scale; stheta = p[12]



		ellip_c = [cmux, cmuy, rwc, rhc, -180/N.pi*ctheta+90, ellipsoid_line_width,    [.0,.0,.0], 'none']
		ellip_s = [smux, smuy, rws, rhs, -180/N.pi*stheta+90, ellipsoid_line_width*2., [.0,.0,.0], 'none']
		dic_rf['theta'] = ctheta
		dic_rf['patch_width'] = patch_width
		dic_rf['mu_xy'] = (cmux, cmuy)

	plots = [dic_rf]


	if not p is None and draw_ellipsoid:
		nones = N.ones(rf_mat.shape)
		annotated_p = ''
		for i,par in enumerate(p_names):
			annotated_p += par + ':' + str(N.round(p[i],2)) + ',  '
			if model=='edog_ext':
				if par == 'cmu_y' or par == 'ccdir_c' or par == 'scdir_c':
					annotated_p += '\n'
			else:
				if (i+1) % 5 == 0:
					annotated_p += '\n'
		
		# print 'center color'
		# print N.round(p[0],2), N.round(p[1],2)
		# print N.round(value_of_rfvector_at(mode, rf, p[0], p[1], patch_width, patch_width**2),2)

		if model=='edog_ext':
			annotated_p += '\n'
			cnt =  __transpose_zero_to_one( N.array([p[5], p[6], p[7]]) ) 
			srn = __transpose_zero_to_one( N.array([p[11], p[12], p[13]]) )
			color = value_of_rfvector_at(mode, rf, p[0], p[1], patch_width, patch_width**2)
			annotated_p += 'center color:' + str(N.round(N.abs(__transpose_zero_to_one( N.array([color[0], color[1], color[2]]) )),2)*255 ) + '\n'
			annotated_p += 'center rgb:' + str(N.round(cnt,2)*255 ) + '\n'
			annotated_p += 'surr rgb:' + str(N.round(srn,2)*255 ) + '\n'


		title = 'params\n' + str(annotated_p) + '\nfit' 
		dic_ellip = {'name':'' if no_title else title, 'value':nones, 'maxmin':False, 'cmap':'Greys', 'balance':False, 'invertaxis':False, 'theta':ctheta, 'patch_width':patch_width, 'drawlines':True}
		dic_ellip['ellipsoid_c'] = ellip_c
		dic_ellip['ellipsoid_s'] = ellip_s	
		plots = plots + [dic_ellip]

	# if True:
	# 	h_dic_curve = {'name':'' if no_title else 'primary axis', 'value':rf, 'mu_xy':(cmux,cmuy), 'theta':ctheta, 'patch_width':patch_width, 'horizontal':False, 'mode':mode, 'drawlines':True}
	# 	plots = plots + [h_dic_curve]
	# 	w_dic_curve = {'name':'' if no_title else 'secondary axis', 'value':rf, 'mu_xy':(cmux,cmuy), 'theta':ctheta, 'patch_width':patch_width, 'horizontal':True, 'mode':mode, 'drawlines':True}
	# 	plots = plots + [w_dic_curve]


	if not rf_reconstr is None:
		rec_mat = convert_rfvector_to_rgbmatrix(rf_reconstr, patch_width, patch_width, mode, swap=False, flip=False)
		rec_mat = __transpose_zero_to_one(rec_mat)
		dic_rec = {'name':'' if no_title else rec_name, 'value':rec_mat, 'maxmin':False, 'cmap':cmap, 'balance':False}
		dic_rec['theta'] = ctheta
		dic_rec['patch_width'] = patch_width
		dic_rec['mu_xy'] = (cmux, cmuy)
	

		plots = plots + [dic_rec]

		if not rf_reconstr_err is None:
			err_mat = convert_rfvector_to_rgbmatrix(rf_reconstr_err, patch_width, patch_width, mode, swap=False, flip=False)
			err_mat = __transpose_zero_to_one(err_mat)
			str_err = str(N.round(N.max(rf_reconstr_err),6))
			dic_err = {'name':'' if no_title else 'error '+str_err, 'value':err_mat, 'maxmin':False, 'cmap':cmap, 'balance':True, 'invertaxis':True}
			plots = plots + [dic_err]

		# if True:
		# 	h_dic_curve = {'name':'' if no_title else 'primary axis', 'value':rf_reconstr, 'mu_xy':(cmux,cmuy), 'theta':ctheta, 'patch_width':patch_width, 'horizontal':False, 'mode':mode, 'drawlines':True}
		# 	plots = plots + [h_dic_curve]
		# 	w_dic_curve = {'name':'' if no_title else 'secondary axis', 'value':rf_reconstr, 'mu_xy':(cmux,cmuy), 'theta':ctheta, 'patch_width':patch_width, 'horizontal':True, 'mode':mode, 'drawlines':True}
		# 	plots = plots + [w_dic_curve]


	write_row_col_fig(plots, 2, 4, outfile, dpi=dpi, alpha=1.0, fontsize=5.5, no_labels=no_title)
	print 'file:', outfile, 'written.'





def write_row_col_fig(
	plots=[], 
	rows=3, 
	cols=3, 
	filepath='./__dbg_row_col_fig.png', 
	dpi=72, 
	fontsize=10, 
	alpha=.75,
	no_labels=False):
	
	def prep_plot(ax, plot_dic):

		try:
			'''plot rf curve - cut in both axes by fit center'''			
			patch_width = plot_dic['patch_width']
			rf = plot_dic['value']

			if plot_dic['horizontal']==True:
				__plot_rF_curve(rf, patch_width, plot_dic['mode'], plot_dic['mu_xy'], plot_dic['theta'], ax)
			else:
				__plot_rF_curve(rf, patch_width, plot_dic['mode'], plot_dic['mu_xy'], plot_dic['theta'], ax, fixed_axis_x=True)

			try:
				if plot_dic['drawlines']:
					from matplotlib.lines import Line2D
					'''draw a line over the plot'''
					theta = plot_dic['theta']/180*N.pi
					patch_width = plot_dic['patch_width']
					pw2 = patch_width/2.
					if plot_dic['horizontal']==True:
						ps = (.4, .55)
						pe = (.4, .75)
						pe = rot(-theta-N.pi/2, pe, ps)
						l = Line2D([ps[0]*pw2,pe[0]*pw2],[ps[1],pe[1]], linewidth=4.0, c='k')
						ax.add_line(l)
					else:
						ps = (.4, .55)
						pe = (.4, .75)
						pe = rot(-theta, pe, ps)
						l = Line2D([ps[0]*pw2,pe[0]*pw2],[ps[1],pe[1]], linewidth=4.0, c='k')
						ax.add_line(l)

			except KeyError:
				pass


		except KeyError:
			'''plot rf weightmatrix'''			
			try:				interp = plot_dic['interp']				
			except KeyError:	interp = 'nearest'
			try:				balance = plot_dic['balance']
			except KeyError:	balance = True
			try:				maxmin  = plot_dic['maxmin']
			except KeyError:	maxmin  = False
			try:				cmap  	= plot_dic['cmap']
			except KeyError:	cmap  	= None
			try:				invert  = plot_dic['invert']
			except KeyError:	invert  = False
			try: 				invertaxis = plot_dic['invertaxis']
			except KeyError: 	invertaxis = False
			__prep_subplot_add_rf(ax, plot_dic['value'], cmap, maxmin, balance, invert, invertaxis, interp, fontsize=fontsize)

			try:
				patch_width = plot_dic['patch_width']
				plt.xlim(0, patch_width-1)
				plt.ylim(0, patch_width-1)
			except KeyError:	pass


			try:			 	draw_center_on_top = plot_dic['draw_center_on_top']
			except KeyError: 	draw_center_on_top = True
			try:			 	par_c = plot_dic['ellipsoid_c']
			except KeyError: 	par_c = None
			try:			 	par_s = plot_dic['ellipsoid_s']
			except KeyError: 	par_s = None
			if draw_center_on_top:
				if par_s!=None:	add_ellipsoid(ax, [par_s[0], par_s[1], par_s[2], par_s[3], par_s[4]], 
											  alpha=alpha, linewidth=par_s[5], 
											  edge_color=par_s[6], face_color=par_s[7])
				if par_c!=None: add_ellipsoid(ax, [par_c[0], par_c[1], par_c[2], par_c[3], par_c[4]], 
											  alpha=alpha, linewidth=par_c[5], 
											  edge_color=par_c[6], face_color=par_c[7])
			else:
				if par_c!=None:	add_ellipsoid(ax, [par_c[0], par_c[1], par_c[2], par_c[3], par_c[4]], 
											  alpha=alpha, linewidth=par_c[5], 
											  edge_color=par_c[6], face_color=par_c[7])
				if par_s!=None:	add_ellipsoid(ax, [par_s[0], par_s[1], par_s[2], par_s[3], par_s[4]], 
											  alpha=alpha, linewidth=par_s[5], 
											  edge_color=par_s[6], face_color=par_s[7])


			try:			 	debug_ellipsoids = plot_dic['debug_ellipsoids']
			except KeyError: 	debug_ellipsoids = None
			if debug_ellipsoids!=None:
				for i in xrange(0,len(debug_ellipsoids)):
					par = debug_ellipsoids[i]
					add_ellipsoid(ax, [par[0], par[1], par[2], par[3], par[4]], 
								  alpha=alpha, linewidth=par[5], 
								  edge_color=par[6], face_color=par[7])

		__prep_subplot(ax, title=plot_dic['name'], show_grid=False)
		if no_labels:
			ax.get_xaxis().set_ticks([])
			ax.get_yaxis().set_ticks([])

	fig = __make_row_col_fig(plots, rows, cols, prep_plot)
	# fig.subplots_adjust(hspace=.05, wspace=.25)
	# fig.subplots_adjust(hspace=.25, wspace=-.25)
	fig.subplots_adjust(hspace=-.55, wspace=0)
	plt.savefig(filepath, bbox_inches='tight', dpi=dpi)
	plt.close(fig)




def __make_row_col_fig(
	dic_of_plots=[], 
	rows=3, 
	cols=3, 
	prepare_subplot=lambda ax,dic:None
	):
	fig = plt.figure()
	for k in range(0, len(dic_of_plots)):
		pos = 100*rows+10*cols+k+1
		dic = dic_of_plots[k]
		ax = fig.add_subplot(pos)
		prepare_subplot(ax, dic)
	return fig



def __plot_rF_curve(
	rf,
	patch_width,
	mode,
	p,
	theta,
	ax,
	fixed_axis_x=False,
	interp_smooth=140,
	):
	from receptivefield import convert_rfvector_to_rgbmatrix
	rf_m = convert_rfvector_to_rgbmatrix(rf, patch_width, patch_width, mode, swap=False, flip=False)
	X = N.arange(0, patch_width, 1)
	X_smooth = N.linspace(0, patch_width-1, interp_smooth)

	real_mu = (N.round(p[0]), N.round(p[1]))
	mu = real_mu[0] if fixed_axis_x else real_mu[1]
	
	if fixed_axis_x:
		coords = [(i, mu) for i in xrange(0, patch_width)]
	else:
		coords = [(mu, i) for i in xrange(0, patch_width)]

	# r_coords = [rot(theta, (c[0],c[1]), real_mu) for c in coords]
	# r_coords = [(round(c[0]),round(c[1])) for c in r_coords]
	r_coords = coords

	def rot_values(channel, rotated_coords):
		# import pdb; pdb.set_trace()
		return N.array([rf_m.T[channel][c[0],c[1]] for c in rotated_coords])

	from scipy.interpolate import interp1d
	def interp(values):
		return interp1d(X, values, kind='cubic')
	
	plt.ylim(-1., 1.)

	if mode=='rgb':
		ax.plot(X_smooth, interp(rot_values(0, r_coords))(X_smooth), color='red', linewidth=2.0)
		ax.plot(X_smooth, interp(rot_values(1, r_coords))(X_smooth), color='green', linewidth=2.0)
		ax.plot(X_smooth, interp(rot_values(2, r_coords))(X_smooth), color='blue', linewidth=2.0)
	elif mode=='rg' or mode=='rg_vs_b':
		ax.plot(X_smooth, interp(rot_values(0, r_coords))(X_smooth), color='red', linewidth=2.0)
		ax.plot(X_smooth, interp(rot_values(1, r_coords))(X_smooth), color='green', linewidth=2.0)
	elif mode=='lum':
		ax.plot(X_smooth, interp(rot_values(0, r_coords))(X_smooth), color='black', linewidth=2.0)

	ax.set_aspect(patch_width/2)
	return mu



def __prep_subplot(
	ax,
	title='None', 
	fontsize=6,
	show_grid=False,
	):
	ax.set_title(title + ' ' + ax.get_title())
	# for m in [ax.title] + ax.get_xticklabels() + ax.get_yticklabels():
	for m in [] + ax.get_xticklabels() + ax.get_yticklabels():
		m.set_fontsize(fontsize)
	ax.title.set_fontsize(fontsize*1.5)	
	if show_grid:
		ax.grid()


def __prep_subplot_add_rf(
	ax,
	a, 
	cmap=None,
	display_maxmin=True,
	norm_color=True,
	invert_data=False,
	invert_axis=False,
	interp='nearest',
	fontsize=6
	):
	if invert_data: 
		a *= -1
	if norm_color:
		a = normalize_color(a)
	'''http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html'''
	im = plt.imshow(a, interpolation=interp, cmap=cmap, vmin=N.min(a), vmax=N.max(a))
	ax.images.append(im)
	if display_maxmin:
		ax.set_title(ax.get_title() + " (" + str(N.round(N.min(a),4)) + "," + str(N.round(N.max(a),4)) + ")")
	if invert_axis:
		ax.invert_yaxis()




def add_rf_subplot(
	fig, 
	pos, 
	a, 
	title='None', 
	fontsize=10, 
	cmap=None,
	display_maxmin=True,
	norm_color=True,
	invert_data=False,
	invert_yaxis=False,
	):
	ax = fig.add_subplot(pos)
	__prep_subplot(ax, title, fontsize, False)
	__prep_subplot_add_rf(ax, a, cmap, display_maxmin, norm_color, invert_data, invert_yaxis)
	return ax

