'''https://en.wikipedia.org/wiki/HSL_and_HSV'''

import numpy as N
import matplotlib.pyplot as plt
import matplotlib.colors as cplt




def manual_rgb_to_hsv(v):
	v = N.array(v)
	'''http://www.rapidtables.com/convert/color/rgb-to-hsv.htm'''
	cmax = N.max(v)
	cmin = N.min(v)
	delta = cmax - cmin
	'''hue'''
	if delta == 0:
		hue = 0
	elif N.isclose(cmax, v[0]):
		hue = 60. * (((v[1] - v[2])/delta) % 6)
	elif N.isclose(cmax, v[1]):
		hue = 60. * (((v[2] - v[0])/delta) + 2.)
	elif N.isclose(cmax, v[2]):
		hue = 60. * (((v[0] - v[1])/delta) + 4.)
	'''sat'''
	if N.isclose(cmax, 0):
		sat = 0
	else:
		sat = delta/cmax
	'''value'''
	val = cmax
	# print v
	# print '\thue', hue, '\tsat', sat, '\tval', val	
	return N.array([hue,sat,val])




# test([1,1,1])

# test([.5,.5,.5])

# test([0,0,0])

# print 

# test([1,0,0])

# test([0,1,0])

# test([0,0,1])

# print 

# test([0,1,1])

# test([1,0,1])

# test([1,1,0])

'''angle = acos(dot(v1,v2) / (|v1| * |v2|))'''


def chroma(r, g, b):
	'''F = max(|gruen-rot|,|blau-rot|,|gruen-blau|)'''
	return N.max([N.max(N.abs(r-g)), N.max(N.abs(b-r)), N.max(N.abs(g-b))])

'''It is observed that for higher values of intensity, 
a saturation of 0.2 differentiates between Hue and Intensity dominance.'''
def sat_threshold(intensitity, intensity_max=1):
	return 1. - ((.8*intensitity)/intensity_max)



def est_color_distr(
	args_file,
	args_odir,
	data_type='tif',
	args_n=1,
	args_all=False,
	args_skew=15
	):


	plt.rcParams['xtick.major.pad']='12'
	plt.rcParams['ytick.major.pad']='12'
	plt.rc('axes', linewidth = 0)


	vis = 13

	from ..base.images import get_paths_of_images, read_image
	from ..base.patches import get_random_patch_from_image

	paths = get_paths_of_images(args_file, data_type, verbose=True)

	assert paths != None and len(paths) != 0

	num_bins = 360.

	
	hsv_bins = N.arange(0,1,float(1/num_bins))

	all_hists = [None] * 10
	for i in xrange(0, len(all_hists)):
		all_hists[i] = N.zeros((int(num_bins)))

	def do_write_image(__n):
		if args_all:
			if __n == 0: 
				return True
			else: 		 
				return False
		else:
			if __n % 500 == 0: 
				return True
			else: 		 	
				return False


	for n in xrange(0,args_n):
		

		img = read_image(paths[N.random.randint(0, len(paths))])

		if args_all:
			hpatch = img
		else:
			patch =	get_random_patch_from_image(img, vis, vis)
			from ..base.patches import prepare_patch
			patch = prepare_patch(patch, mode='rgb', 
				subtract_mean=True, chance_to_sawp_axes=0)

			from ..base.receptivefield import convert_rfvector_to_rgbmatrix
			hpatch = convert_rfvector_to_rgbmatrix(patch, vis, vis, 'rgb', swap=False, flip=False)
			hpatch -= hpatch.min()

		hsvpatch = cplt.rgb_to_hsv(hpatch)

		for i in xrange(0, hsvpatch.T[0].shape[0]):
			for j in xrange(0, hsvpatch.T[0].shape[1]):
				saturation = hsvpatch.T[1][i][j]
				intensity = hsvpatch.T[2][i][j]

				hue = hsvpatch.T[0][i][j]*359

				# skew = .5 * args_skew + .5 * (intensity) * args_skew
				skew = N.random.normal()*args_skew
				hue += ((saturation)-.5)*skew
				hue = hue % 360
				if hue == 360: hue = 0

				hist_id = int(N.round(intensity * 10))-1
				# import pdb; pdb.set_trace()
				all_hists[hist_id][int(hue)] += 1

				
		if do_write_image(n):

			span = hsv_bins.max()-hsv_bins.min()
			cmH = plt.cm.get_cmap('hsv')
			C = [cmH(((x-hsv_bins.min())/span)) for x in hsv_bins]
			# cmB = plt.cm.get_cmap('binary')
			# B = [cmB(1-((x-hsv_bins.min())/span)) for x in hsv_bins]

			fig = plt.figure()
			ax = fig.add_subplot(111)

			all_max = 0
			for i, hist in enumerate(all_hists):
				ax.bar(hsv_bins, hist, 1/num_bins, color=C, linewidth=0, bottom=all_max)
				all_max += N.max(hist)

			ax.set_xticks(N.arange(0,1.1,1/6.))
			ax.set_yticks([])
			

			plt.savefig('./test.png', bbox_inches='tight', dpi=144)
			plt.close(fig)

	print 





def est_color_distr_rgb(
	args_file,
	args_odir,
	data_type='tif',
	args_n=1,
	args_all=False,
	args_skew=15
	):

	plt.rcParams['xtick.major.pad']='12'
	plt.rcParams['ytick.major.pad']='12'
	plt.rc('axes', linewidth = 0)


	vis = 13

	from ..base.images import get_paths_of_images, read_image
	from ..base.patches import get_random_patch_from_image

	paths = get_paths_of_images(args_file, data_type, verbose=True)

	assert paths != None and len(paths) != 0

	chm = vis**2
	bins = N.arange(-1.,1.,1/360.)

	r_hist = N.zeros((bins.shape[0]-1))
	g_hist = N.zeros((bins.shape[0]-1))
	b_hist = N.zeros((bins.shape[0]-1))

	# hsv_s_hist = N.zeros((hsv_bins.shape[0]-1))
	# hsv_v_hist = N.zeros((hsv_bins.shape[0]-1))

	for n in xrange(0,args_n):

		img = read_image(paths[N.random.randint(0, len(paths))])

		patch =	get_random_patch_from_image(img, vis, vis)

		from ..base.patches import prepare_patch
		patch = prepare_patch(patch, mode='rgb', 
			subtract_mean=True, chance_to_sawp_axes=0)

		rh = N.histogram(patch[0:chm], bins=bins)
		r_hist = r_hist + rh[0]

		gh = N.histogram(patch[chm:2*chm], bins=bins)
		g_hist = g_hist + gh[0]

		bh = N.histogram(patch[2*chm:3*chm], bins=bins)
		b_hist = b_hist + bh[0]





	fig = plt.figure()
	ax = fig.add_subplot(111)


	rmax = N.max(r_hist)
	gmax = N.max(g_hist)

	ax.bar(bins[:-1], r_hist, 1/360., color='r', linewidth=0, alpha=1., bottom=rmax+gmax)
	ax.bar(bins[:-1], g_hist, 1/360., color='g', linewidth=0, alpha=1., bottom=rmax)
	ax.bar(bins[:-1], b_hist, 1/360., color='b', linewidth=0, alpha=1.)
	

	plt.xlim(-1,1)

	plt.savefig('./test.png', bbox_inches='tight', dpi=144)
	plt.close(fig)







