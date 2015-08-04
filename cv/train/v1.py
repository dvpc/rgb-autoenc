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




def train_v1(
	**kwargs
	):
	from ..base.weightmatrix import load_2_as_dict
	rgcld_args = load_2_as_dict(
		kwargs['rgcfile'], dont_load_matrix=True, verbose=True)
	kwargs['mode'] = rgcld_args['mode']
	kwargs['vis'] = rgcld_args['vis']
	kwargs['hid'] = rgcld_args['hid']

	if kwargs['v1file'] is None:
		epochs_done = 0
	else:
		v1ld_args = load_2_as_dict(
			kwargs['v1file'], dont_load_matrix=True, verbose=True)
		# mode is alway lum for v1!!!
		kwargs['hid1'] = v1ld_args['hid1']
		kwargs['hid2'] = v1ld_args['hid2']
		kwargs['a'] = v1ld_args['a']
		kwargs['b'] = v1ld_args['b']
		kwargs['d'] = v1ld_args['d']
		kwargs['amp'] = v1ld_args['amp']
		kwargs['mom'] = v1ld_args['mom']
		epochs_done = v1ld_args['epochs_done']
		if kwargs['lr'] == 0.0555:
			kwargs['lr'] = v1ld_args['lr']



	from ..base import v1_filename_str
	map_outstr = v1_filename_str(**kwargs)

	from ..train import import_functions_by_mode
	exp_f, expcolor_f = \
		import_functions_by_mode(kwargs['mode'])

	from ..base import make_working_dir
	writer_dir = make_working_dir(**kwargs)

	'''copy map file into writer_dir'''
	import os
	rgcfilename = os.path.split(kwargs['rgcfile'])[-1]
	rgcfilepath = os.path.join(writer_dir, rgcfilename)

	if not os.path.exists(rgcfilepath):
		import shutil
		shutil.copy2(kwargs['rgcfile'], rgcfilepath)


	alg_args = dict({
		'vis': kwargs['vis'],
		'hid1': kwargs['hid'],
		'hid2': kwargs['hid2'],
		'lr': kwargs['lr'],
		'a': kwargs['a'],
		'b': kwargs['b'],
		'd': kwargs['d'],
		'amp' : kwargs['amp'],
		'mom' : kwargs['mom'],
		})

	procs = []

	from ..train import patcher_filepool_proc as patcher_proc
	from ..train import writer_proc
	from ..train import trainer_proc
	from ..train import train_skeleton
	from ..train import __clean_func
	return train_skeleton(procs,
		num_patcher=kwargs['ncpu'],
		spawn_process_for_writer=kwargs['spawn_writer'],
		p_proc=patcher_proc,
		p_args={'data_dir': kwargs['datadir'], 
				'data_type': kwargs['dtype'],
				'num_patches_per_image': kwargs['npt'],
				'vis': kwargs['vis'],
				'mode': kwargs['mode'],
				'ptp': kwargs['ptp'],

				'patcher_filepool_n_open': kwargs['patcher_filepool_n_open'],
				'patcher_filepool_update_s': kwargs['patcher_filepool_update_s'],
				'patcher_filepool_update_nfiles': kwargs['patcher_filepool_update_nfiles'],
				'batch_samples_n': kwargs['batch_samples_n'],
				'exclude_below_avg': True,
				'patch_chance_to_sawp_axes': kwargs['patch_chance_to_sawp_axes'], 
			},
		w_proc=writer_proc,
		w_args=dict({'which': kwargs['which'],
				'writer_func': __writeout_func,
				'sequence': kwargs['outseq'],
				'mode': kwargs['mode'],
				'exp_f': exp_f,
				'expcolor_f': expcolor_f,
				'odir': kwargs['odir'],
				'tdir': kwargs['tdir'],
				'wdir': writer_dir,
				'map_outstr': map_outstr,
				'epochs_done': epochs_done,
				'writer_logfile': kwargs['writer_logfile'],
			}.items() + alg_args.items()),
		t_proc=trainer_proc,
		t_args=dict({'trainer_func': 
					__train_func_theano if kwargs['th'] \
					else __train_func,
				'out_each_n': kwargs['outn'],
				'max_epochs': kwargs['maxep'],
				'mode': kwargs['mode'],
				'rgcfile': kwargs['rgcfile'],
				'v1file': kwargs['v1file'],

				'batch_samples_n': kwargs['batch_samples_n'],
			}.items() + alg_args.items()),
		clean_f=__clean_func,
		clean_args=dict({
				'which': kwargs['which'],
				'map_outstr': map_outstr,
				'wdir': writer_dir,
				'sequence': kwargs['outseq'],
			}),
		)	




def __writeout_func(
	odict,
	**kwargs
	):
	wdir = kwargs['wdir']
	tdir = kwargs['tdir']
	exp_f = kwargs['exp_f']
	expcolor_f = kwargs['expcolor_f']
	from ..base import num_channel_of
	vis_ch = num_channel_of(kwargs['mode'])
	vis = kwargs['vis']
	hid1 = kwargs['hid1']
	hid2 = kwargs['hid2']
	
	W = odict['W']
	expcolor_f (W, vis*vis_ch, vis, tdir+'/obs_W_0_1.pnm', hid1, hid1)
	expcolor_f (odict['x'], vis*vis_ch, vis, tdir+'/obs_X_0.pnm')
	W_v1 = odict['W_v1']	
	exp_f (W_v1, hid1, hid1, tdir+'/obs_W_1_2.pnm', hid2, hid2)
	exp_f (odict['x1'], hid1, hid1, tdir+'/obs_X_1.pgm')
	exp_f (odict['y'], hid2, hid2, tdir+'/obs_Y_2.pgm')
	exp_f (odict['z'], hid2, hid2, tdir+'/obs_Z_2.pgm')
	exp_f (odict['x1_rec'], hid1, hid1, tdir+'/obs_R_1.pgm')
	exp_f (odict['err'], hid1, hid1, tdir+'/obs_E_1.pgm')
	W_2_0 = N.dot(W_v1, W)
	expcolor_f (W_2_0, vis*vis_ch, vis, tdir+'/obs_W_0_2.pnm', hid2, hid2)

	from ..base.weightmatrix import save_2
	save_2(
		filepath=wdir+'/'+kwargs['map_outstr']+'.map',
		W=W_v1, 
		W_args={
			'epochs_done': kwargs['epochs_done']+odict['epochs'],
			'mode' : 'lum',
			'hid1': kwargs['hid1'],
			'hid2': kwargs['hid2'],
			'lr': kwargs['lr'],			
			'a': kwargs['a'],
			'b': kwargs['b'],
			'd': kwargs['d'],
			'amp' : kwargs['amp'],
			'mom' : kwargs['mom'],
			},
		version=2)
	import os
	if kwargs['sequence']: 
		from ..base import pad_sequence_num
		seqstr = '_' + \
			pad_sequence_num(kwargs['epochs_done']+odict['epochs'])
		write_dir = os.path.join(wdir,'img')
	else:        
		seqstr = ''
		write_dir = wdir
	filename = os.path.join(write_dir,'v1'+seqstr+'.png')
	from ..base.plots import save_w_as_image
	save_w_as_image(W_2_0, vis, vis, hid2, hid2,
		mode=kwargs['mode'], 
		outfile=filename, 
		verbose=False)



def __train_func(
	patchq,
	outq,
	W, 
	W_v1, 
	**kwargs	
	):	
	import time
	last_time = time.time()

	lr = kwargs['lr']
	a = kwargs['a']
	b = kwargs['b']
	decay = kwargs['d']
	amp = kwargs['amp']
	b_sq = b**2	
	def olshausen_tf(x):
		return b * (x - a * x / (1.0 + b_sq * x**2))

	for n in xrange(kwargs['max_epochs']):
		batch = patchq.get()
		for batch_index in xrange(kwargs['batch_samples_n']):
			x = batch[batch_index]
			x1 = N.dot(W, x)*amp
			y = N.dot(W_v1, x1)
			z = olshausen_tf(y)
			x1_rec = N.dot(W_v1.T, z)
			err = x1 - x1_rec
			dW = lr * N.outer(z, err)
			W_v1 += dW
			W_v1 -= lr * decay * W_v1

		if n % kwargs['out_each_n'] == 0:
			if not outq.full():
				outq.put({'W':W, 'W_v1':W_v1,
						  'x':x, 'x1':x1, 'y':y, 'z':z, 
						  'x1_rec':x1_rec, 'err':err, 
						  'epochs':n*kwargs['batch_samples_n'], 
						  'elapsed':time.time() - last_time}, 
						  block=False)
				last_time = time.time()



def __train_func_theano(
	patchq,
	outq,
	W, 
	W_v1, 
	**kwargs	
	):
	import time
	last_time = time.time()

	lr = kwargs['lr']
	a = kwargs['a']
	b = kwargs['b']
	decay = kwargs['d']
	amp = kwargs['amp']
	mom = kwargs['mom']
	b_sq = b**2	

	import theano
	from theano import tensor as T
	index = T.lscalar()    # index to a [mini]batch
	if kwargs['mode']=='lum': 
		zeros = N.zeros((kwargs['batch_samples_n'],kwargs['vis']**2))
	elif kwargs['mode']=='rgb': 
		zeros = N.zeros((kwargs['batch_samples_n'],kwargs['vis']**2*3))
	else:
		zeros = N.zeros((kwargs['batch_samples_n'],kwargs['vis']**2*2))
	batch_shared = theano.shared(zeros, borrow=True)



	th_W = theano.shared(name='wrg', borrow=True, value=W)
	th_W_v1 = theano.shared(name='wbu', borrow=True, value=W_v1)
	th_M_v1 = theano.shared(name='m', borrow=True, value=N.zeros(W_v1.shape)) # momentum

	th_x = T.dmatrix('x')
	th_x1 = T.dot(th_W, th_x[0])*T.constant(amp)
	th_y = T.dot(th_W_v1, th_x1)
	th_z = T.constant(b) * (th_y - T.constant(a) * th_y / (1.0 + T.constant(b_sq) * th_y**2))
	th_x1_rec = T.dot(th_W_v1.T, th_z)
	th_err = th_x1 - th_x1_rec
	th_dW = T.constant(lr) * T.outer(th_z, th_err)
	th_decW = th_dW - T.constant(lr) * T.constant(decay) * th_W_v1


	step_batch = theano.function(
		inputs=[index],
		# updates=((th_W_v1, th_W_v1 + th_decW),), 
		updates=((th_W_v1, th_W_v1 + mom*th_M_v1 + th_decW), (th_M_v1, th_dW)), 
		outputs=[th_x1, th_y, th_z, th_x1_rec, th_err, th_W_v1],
		givens={th_x: batch_shared[index:index+1],}
		)

	for n in xrange(kwargs['max_epochs']):
		batch = patchq.get()

		batch_shared.set_value(batch, borrow=True)
		for batch_index in xrange(kwargs['batch_samples_n']):
			x1, y, z, x1_rec, err, W_v1 = step_batch(batch_index)

		if n % kwargs['out_each_n'] == 0:
			if not outq.full():
				x = batch[0]
				outq.put({'W':W, 'W_v1':W_v1,
						  'x':x, 'x1':x1, 'y':y, 'z':z, 
						  'x1_rec':x1_rec, 'err':err, 
						  'epochs':n*kwargs['batch_samples_n'], 
						  'elapsed':time.time() - last_time}, 
						  block=False)
				last_time = time.time()


