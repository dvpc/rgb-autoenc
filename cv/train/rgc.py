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




def train_rgc(
	**kwargs
	):
	if kwargs['rgcfile'] is None:
		epochs_done = 0
	else:
		from ..base.weightmatrix import load_2_as_dict
		wld_args = load_2_as_dict(
			kwargs['rgcfile'], dont_load_matrix=True, verbose=True)
		kwargs['mode'] = wld_args['mode']
		kwargs['vis'] = wld_args['vis']
		kwargs['hid'] = wld_args['hid']
		if kwargs['k'] == 1.1111e-5:
			kwargs['k'] = wld_args['k']
		kwargs['k'] = wld_args['k']
		kwargs['p'] = wld_args['p']
		kwargs['clip'] = wld_args['clip']
		epochs_done = wld_args['epochs_done']
		if kwargs['lr'] == 0.0555:
			kwargs['lr'] = wld_args['lr']
		try:
			kwargs['ptp'] = wld_args['ptp']
		except KeyError:
			kwargs['ptp'] = False

	from ..base import rgc_filename_str
	map_outstr = rgc_filename_str(**kwargs)

	from ..train import import_functions_by_mode
	exp_f, expcolor_f = \
		import_functions_by_mode(kwargs['mode'])

	from ..base import make_working_dir
	writer_dir = make_working_dir(**kwargs)



	alg_args = dict({
		'vis': kwargs['vis'],
		'hid': kwargs['hid'],
		'lr': kwargs['lr'],
		'clip': kwargs['clip'],
		'k': kwargs['k'],
		'p': kwargs['p'],
		'ptp': kwargs['ptp'],
		'leaky': kwargs['leaky'],
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
				'exclude_below_avg': False,
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
	hid = kwargs['hid']

	W = odict['W']
	expcolor_f (W, vis*vis_ch, vis, tdir+'/obs_W_0_1.pnm', hid, hid)
	expcolor_f (odict['x'], vis*vis_ch, vis, tdir+'/obs_X_0.pnm')
	exp_f (odict['y'], hid, hid, tdir+'/obs_Y_1.pgm')
	expcolor_f (odict['z'], vis*vis_ch, vis, tdir+'/obs_Z_0.pnm')
	expcolor_f (odict['err'], vis*vis_ch, vis, tdir+'/obs_E_0.pnm')

	from numpy import shape, zeros, allclose
	Wconstr = zeros(hid*hid, dtype=float)
	for i in range(0, shape(W)[0]):
		isclose = allclose(W[i], zeros(W[i].shape), atol=1e-02)
		if isclose:	Wconstr[i] = 0.0
		else: 			Wconstr[i] = 1.0
	exp_f (Wconstr, hid, hid, tdir+'/obs_C_1.pgm')
	
	from ..base.weightmatrix import save_2
	save_2(
		filepath=wdir+'/'+kwargs['map_outstr']+'.map',
		W=W, 
		W_args={
			'epochs_done': kwargs['epochs_done']+odict['epochs'],
			'mode' : kwargs['mode'],
			'vis': kwargs['vis'],
			'hid': kwargs['hid'],
			'lr': kwargs['lr'],			
			'clip': kwargs['clip'],
			'k': kwargs['k'],
			'p': kwargs['p'],
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
	filename = os.path.join(write_dir,'rgc'+seqstr+'.png')
	from ..base.plots import save_w_as_image
	save_w_as_image(W, vis, vis, hid, hid, 
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
	
	clip = kwargs['clip']
	leaky = kwargs['leaky']
	lr = kwargs['lr']
	k = kwargs['k']
	p = kwargs['p']

	for n in xrange(kwargs['max_epochs']):
		batch = patchq.get()
		for batch_index in xrange(kwargs['batch_samples_n']):
			x = batch[batch_index]
			if clip:	y = N.clip(N.dot(W, x), leaky, N.inf)
			else:		y = N.dot(W, x)		
			z = N.dot(W.T, y)
			err = x - z
			dW = lr * N.outer(y, err)
			W += dW
			for i in range(0, N.shape(W)[0]):
				W[i] += lr * -k * N.copysign(1, W[i]) * N.abs(W[i])**p
		if n % kwargs['out_each_n'] == 0:
			if not outq.full():
				outq.put({'W':W, 'x':x, 'y':y, 'z':z, 
						  'err':err, 
						  'epochs':n*kwargs['batch_samples_n'], 
						  'elapsed':time.time() - last_time}, 
						  block=False)
				last_time = time.time()



def __train_func_theano(
	patchq,
	outq,
	WW, 
	W_v1, 
	**kwargs
	):
	import time
	last_time = time.time()

	clip = kwargs['clip']
	leaky = kwargs['leaky']
	lr = kwargs['lr']
	k = kwargs['k']
	p = kwargs['p']

	'''theano model'''
	import theano
	from theano import tensor as T
	cfn=lambda x: \
		x + T.constant(lr) * \
		T.constant(-k) * T.sgn(x) * \
		T.abs_(x)**T.constant(p)
	m = T.dmatrix('m')
	res, upd = theano.scan(
		fn=cfn, 
		outputs_info=[None], 
		n_steps=m.shape[0], 
		sequences=m)
	constr = theano.function(
		inputs=[theano.In(m, borrow=True)], 
		updates=upd, 
		mode=theano.Mode(linker='cvm'), 
		outputs=theano.Out(res, borrow=True),)
	W = theano.shared(name='W', borrow=True, value=WW)
	x = T.dvector('x')
	if clip: y = T.clip(T.dot(W, x), T.constant(leaky), N.inf)
	else:	 y = T.dot(W, x)
	z = T.dot(W.T, y)
	err = x - z
	dW = T.constant(lr) * T.outer(y, err)
	train_step = theano.function(
		inputs=[x], 
		outputs=[theano.Out(err, borrow=True),
				 theano.Out(y, borrow=True),
				 theano.Out(z, borrow=True),],
		updates=( (W, W + dW), ), 
		allow_input_downcast=True)
	'''end theano model'''	

	for n in xrange(kwargs['max_epochs']):
		batch = patchq.get()
		for batch_index in xrange(kwargs['batch_samples_n']):
			x = batch[batch_index]
			err, y, z = train_step(x)
			WW = W.get_value(borrow=True, return_internal_type=True)
			WW = constr(WW)
			W.set_value(WW, borrow=True)
		if n % kwargs['out_each_n'] == 0: 
			if not outq.full():
				outq.put({'W':WW, 'x':x, 'y':y, 'z':z, 
						  'err':err, 
						  'epochs':n*kwargs['batch_samples_n'], 
						  'elapsed':time.time() - last_time}, 
						  block=False)
				last_time = time.time()








