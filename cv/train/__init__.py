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

def import_functions_by_mode(
	mode,
	):
	if mode == 'lum':
		from ..base.exptiles import exporttiles as expcolor_f
	elif mode == 'rg':
		from ..base.exptiles import exporttiles_rg as expcolor_f
	elif mode == 'rg_vs_b':	
		from ..base.exptiles import exporttiles_rg_vs_b as expcolor_f
	elif mode == 'rgb':
		from ..base.exptiles import exporttiles_rgb as expcolor_f
	from ..base.exptiles import exporttiles as exp_f		
	return exp_f, expcolor_f




def start_training_from_args(
	args,
	spawn_writer_process=False,
	writer_logfile=None,
	
	patcher_filepool_n_open=100,
	patcher_filepool_update_s=2,
	patcher_filepool_update_nfiles=5,

	):
	if args.which=='rgc':
		from cv.train.rgc import train_rgc
		return train_rgc(
				which=args.which,
				spawn_writer=spawn_writer_process,
				writer_logfile=writer_logfile,
				odir=args.odir, 
				tdir=args.tdir, 
				datadir=args.datadir,
				dtype=args.dtype,
				maxep=args.maxep,
				outn=args.outn,
				outseq=args.outseq,
				ncpu=args.ncpu, 
				npt=args.npt,
				ptp=args.ptp,
				th=args.th,
				mode=args.mode,
				patcher_filepool_n_open=patcher_filepool_n_open,
				patcher_filepool_update_s=patcher_filepool_update_s,
				patcher_filepool_update_nfiles=patcher_filepool_update_nfiles,
				batch_samples_n=args.btchn,
				patch_chance_to_sawp_axes=args.swp, 

				v1file=None,
				rgcfile=args.rgcfile,
				lr=args.lr,
				vis=args.vis,
				hid=args.hid,
				clip=args.clip,
				leaky=args.leaky,
				k=args.k,
				p=args.p,
				trsh=args.trsh,

				rdlrint=args.rdlrint,
				rdlrfac=args.rdlrfac,
				overwlr=args.overwlr,
			)
	elif args.which=='drgc':
		from cv.train.drgc import train_drgc
		return train_drgc(
				which=args.which,
				spawn_writer=spawn_writer_process,
				writer_logfile=writer_logfile,
				odir=args.odir, 
				tdir=args.tdir, 
				datadir=args.datadir,
				dtype=args.dtype,
				maxep=args.maxep,
				outn=args.outn,
				outseq=args.outseq,
				ncpu=args.ncpu, 
				npt=args.npt,
				ptp=args.ptp,
				th=args.th,
				mode=args.mode,
				patcher_filepool_n_open=patcher_filepool_n_open,
				patcher_filepool_update_s=patcher_filepool_update_s,
				patcher_filepool_update_nfiles=patcher_filepool_update_nfiles,
				batch_samples_n=args.btchn,
				patch_chance_to_sawp_axes=args.swp, 

				v1file=None,
				rgcfile=args.rgcfile,
				lr=args.lr,
				vis=args.vis,
				hid=args.hid,
				clip=args.clip,
				leaky=args.leaky,
				k=args.k,
				p=args.p,
				corr=args.corr,
			)
	elif args.which=='v1':
		from cv.train.v1 import train_v1
		return train_v1(
				which=args.which,
				spawn_writer=spawn_writer_process,
				writer_logfile=writer_logfile,
				odir=args.odir, 
				tdir=args.tdir, 
				datadir=args.datadir,
				dtype=args.dtype,
				maxep=args.maxep,
				outn=args.outn,
				outseq=args.outseq,
				ncpu=args.ncpu, 
				npt=args.npt,
				ptp=args.ptp,
				th=args.th,
				mode=args.mode,
				patcher_filepool_n_open=patcher_filepool_n_open,
				patcher_filepool_update_s=patcher_filepool_update_s,
				patcher_filepool_update_nfiles=patcher_filepool_update_nfiles,
				batch_samples_n=args.btchn,
				patch_chance_to_sawp_axes=args.swp, 

				v1file=args.v1file,
				rgcfile=args.rgcfile,
				lr=args.lr,
				hid2=args.hid2,
				a=args.a,
				b=args.b,
				d=args.decay,
				amp=args.amp,
				mom=args.mom,
			)
	elif args.which=='rgcv1':
		raise NotImplementedError







def train_skeleton(
	procs,
	num_patcher=1,
	spawn_process_for_writer=False,
	t_proc=lambda pq,oq,:None,
	t_args=(),	
	p_proc=lambda pq,:None,
	p_args=(),	
	w_proc=lambda oq,:None,
	w_args=(),
	clean_f=lambda oq,:None,
	clean_args=(),
	):
	def clean(procs):
		clean_f(**clean_args)
		for p in procs:	p.terminate()
	import atexit
	atexit.register(clean, procs)
	import multiprocessing
	'''patch queue'''
	pq = multiprocessing.Queue(100)
	'''out queue'''
	oq = multiprocessing.Queue(0)
	for i in xrange(0,num_patcher):
		pt = multiprocessing.Process(target=p_proc, args=(pq,), kwargs=p_args)
		procs.append(pt)
	trnr = multiprocessing.Process(target=t_proc, args=(pq,oq,), kwargs=t_args)
	procs.append(trnr)
	if spawn_process_for_writer:
		wrtr = multiprocessing.Process(target=w_proc, args=(oq,), kwargs=w_args)
		procs.append(wrtr)
	'''start processes'''
	for p in procs:	
		p.start()
	'''execute writer in local process'''
	if not spawn_process_for_writer:
		w_proc(oq, **w_args)
	return procs




def writer_proc(
	outpt_queue,
	**kwargs
	):
	writer_dir = kwargs['wdir']
	writer_func = kwargs['writer_func']
	import os	
	logfile = os.path.join(writer_dir,'train.log')

	from ..base import create_dir
	create_dir(kwargs['tdir'])
	if kwargs['sequence']:
		create_dir(writer_dir+'/img', remove_if_exists=True)

	from ..base import write_param_info_file
	write_param_info_file(writer_dir, **kwargs)

	epochs_done = kwargs['epochs_done']

	outdict = None
	from ..base import print_elapsed	
	total_elapsed = 0
	while True:
		outdict = outpt_queue.get()
		writer_func(outdict, **kwargs)

		err_sq = 0.5 * outdict['err']**2
		err_mn = err_sq.mean()
		err_str = str(err_mn)
		try:
			err1_sq = 0.5 * outdict['err1']**2
			err1_mn = err1_sq.mean()
			err_str = '(rgc) ' + str(N.round(err_mn,6)) + ' (v1) ' + str(N.round(err1_mn,6))
		except KeyError:
			pass
		try:
			lrstr = outdict['lr']
		except KeyError:
			lrstr = ''
		total_elapsed += outdict['elapsed']
		out_str = 'n: ' + str(epochs_done+outdict['epochs']) + \
			  	  '    \terr: '+ err_str + \
				  '    \telapsed ' + str(outdict['elapsed']) + \
				  '	   \ttotal ' + str(print_elapsed(total_elapsed)) + \
				  '    \tlr ' + str(lrstr)
		if kwargs['writer_logfile']:
			f = open(kwargs['writer_logfile'], 'a')
			f.write(out_str+'\n')
			f.close()
		else:
			print out_str
		'''also write last line into wdir'''
		f = open(logfile,'a')
		f.write(out_str+'\n')
		f.close()




def trainer_proc(
	patch_queue, 
	outpt_queue,
	trainer_func=lambda oq, wrgc, wv1, **args:None,
	**kwargs
	):
	if kwargs['rgcfile'] is None:
		from ..base.weightmatrix import init_weight_matrix
		W_rgc = init_weight_matrix(kwargs['mode'], kwargs['vis'], kwargs['hid'])
	else:
		from ..base.weightmatrix import load_2
		W_rgc = load_2(kwargs['rgcfile'], dont_load_matrix=False, verbose=False)

	if kwargs['v1file'] is None:
		try:
			from ..base.weightmatrix import init_weight_matrix
			W_v1 = init_weight_matrix('lum', kwargs['hid1'], kwargs['hid2'], stddev=0.3)
		except KeyError:
			W_v1 = None
	else:
		from ..base.weightmatrix import load_2
		W_v1 = load_2(kwargs['v1file'], dont_load_matrix=False, verbose=False)

	trainer_func(patch_queue, outpt_queue, W_rgc, W_v1, **kwargs)





def patcher_proc(
	patch_queue,
	**kwargs
	):
	from ..base.images import get_paths_of_images, read_image
	from ..base.patches import prepare_patch
	from ..base.patches import get_random_patch_from_image
	paths = get_paths_of_images(kwargs['data_dir'], kwargs['data_type'], verbose=False)
	img = None
	dont_subtract_mean = kwargs['ptp']
	while True:
		img = read_image(
			paths[N.random.randint(0, len(paths))])
		for x in xrange(0,kwargs['num_patches_per_image']):
			patch_queue.put( 
				prepare_patch(
					get_random_patch_from_image(img, kwargs['vis'], kwargs['vis']), 
					mode=kwargs['mode'], 
					subtract_mean=not dont_subtract_mean,
					chance_to_sawp_axes=kwargs['patch_chance_to_sawp_axes']
					)
				)
		del img





def patcher_filepool_proc(
	patch_queue,
	**kwargs
	):
	from ..base import sum_sizeof
	from ..base.images import get_paths_of_images, read_image
	from ..base.patches import prepare_patch
	from ..base.patches import get_random_patch_from_image
	paths = get_paths_of_images(kwargs['data_dir'], kwargs['data_type'], verbose=False)

	from datetime import datetime, timedelta

	files = []
	for i in xrange(0,kwargs['patcher_filepool_n_open']):
		'''open a new file and add it to the pool'''
		img = read_image(paths[N.random.randint(0, len(paths))])
		files.append(img)

	# TODO make it an option / its own command
	# from ..base.patches import get_avergae_variance_over_images
	# avg_variance = get_avergae_variance_over_images(paths)
	avg_variance = 0.0589692271522

	upd_int_t = timedelta(seconds=kwargs['patcher_filepool_update_s'])
	st_t = datetime.now()
	upd_t = st_t + upd_int_t

	while True:
		'''replace files in pool'''
		if datetime.now() > upd_t:
			upd_t = datetime.now() + upd_int_t
			for i in xrange(0,kwargs['patcher_filepool_update_nfiles']):
				rnd_id = N.random.randint(0, kwargs['patcher_filepool_n_open'])
				# print 'replacing', rnd_id
				files[rnd_id] = read_image(paths[N.random.randint(0, len(paths))])

		'''get n patches from images in pool'''
		batch = []
		batch_size_b = 0
		while len(batch) < kwargs['batch_samples_n']:
			rnd_id = N.random.randint(0, kwargs['patcher_filepool_n_open'])
			patch =	get_random_patch_from_image(files[rnd_id], kwargs['vis'], kwargs['vis'])
			if not kwargs['exclude_below_avg']:
				add_patch = True
			else:
				if N.var(patch) / avg_variance > .1:
					add_patch = False
					pass
					'''discard patch as it is below 10 % of average imgage variance
					   see olshausen.'''
				else:
					add_patch = True
			if add_patch:
				batch.append( prepare_patch(patch, 
					mode=kwargs['mode'], 
					subtract_mean=not kwargs['ptp'], 
					chance_to_sawp_axes=kwargs['patch_chance_to_sawp_axes']) )
				batch_size_b += sum_sizeof(patch)			
		# print 'batch size b', batch_size_b, ' #', len(batch)
		patch_queue.put( N.copy(batch) )






def __clean_func(
	**kwargs
	):
	import os
	logfile = os.path.join(kwargs['wdir'],'train.log')
	f = open(logfile,'a')
	f.write('cleaning up...\n')
	f.write('done.\n')
	f.close()




