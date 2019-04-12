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



def f_to_min(
	p, 
	f,
	rf,
	channel_w, 
	patch_w
	):
	sqerr = (rf - f(p, channel_w, patch_w, *N.indices(rf.shape)))**2
	return N.sum(sqerr)



from scipy import optimize
''' SLSQP  Sequential Least Squares Programming
	http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html'''
def fit_slsqp(
	f,	
	rf, 
	init_p, 
	bounds_p, 
	channel_w, 
	patch_w,
	maxiter=10000,
	constraints=()
	):
	res = optimize.minimize(f_to_min, init_p, 
		args=(f, rf, channel_w, patch_w), 
		method='SLSQP', options={'maxiter':maxiter}, 
		bounds=bounds_p, constraints=constraints)
	return res.x, res.fun



def bestfit_skel(
	perm_p_f=lambda x,y:x,
	fit_f=lambda x:x,
	debug_f=lambda x,xy,y,z:x,
	init_p=[],
	num_attempts=10,
	debug_onlybestfit=True,
	return_error=False,
	):
	best_p = []
	best_fun = float("inf")	
	best_idx = 0
	if debug_onlybestfit: __debug_f = lambda x,xy,y,z:x
	else:				  __debug_f = debug_f
	for n in xrange(0,num_attempts):
		p = perm_p_f(n, init_p)
		p, min_fun = fit_f(p)
		if min_fun < best_fun:
			best_idx = n
			best_p = p
			best_fun = min_fun
			__debug_f(n, best_p, best_fun, str(n))
	debug_f(best_idx, best_p, best_fun, 'best' + str(best_idx))
	if return_error:	return best_p, best_fun
	else:				return best_p


def multiproc_fit2(
	data=[], 
	func=lambda chunk, *args:(),
	args=(), 
	nprocs=2, 
	verbose=False
	):
	import sys, multiprocessing
	in_q = multiprocessing.Queue()
	out_q = multiprocessing.Queue()
	completed = multiprocessing.Value('i', 0)
	total = multiprocessing.Value('i', len(data))
	
	def worker(iq, oq, num_completed, num_total, worker_id=-1):
		while num_completed.value < num_total.value:
			try:
				dic = iq.get(timeout=100)
				out_q.put( func(dic, *args) )
				num_completed.value += 1
				sys.stdout.write(
					'\rworker ' +  str(worker_id) + \
					' (' + str(num_completed.value) + \
					' of ' + str(num_total.value) + ') '
					' computes RF # ' + str(dic['n']) + 10*' ' )
				sys.stdout.flush()				
			except Exception:
				break
		if verbose: print '\r worker ' + str(worker_id) + ' completed                   '

	for item in data:
		in_q.put(item)

	procs = []
	def clean(procs):
		for p in procs:	p.terminate()
	import atexit
	atexit.register(clean, procs)

	for i in range(nprocs):
		if verbose: print '\r worker ' + str(i) + ' started'
		p = multiprocessing.Process(
			target=worker, 
			args=(in_q, out_q, completed, total, i))
		procs.append(p)
		p.start()
	
	resultdict = {}	
	import time
	num_attempts = 0
	while True:
		try:
			dic = out_q.get(timeout=100)
		except Exception:
			num_attempts += 1
			dic = None
		if dic is not None:
			key = dic['n']
			resultdict[key] = dic
		if completed.value == total.value or num_attempts == 2:
			clean(procs)
			break
		time.sleep(.5)
	return resultdict


def multiproc_fit(
	data=[], 
	func=lambda chunk, *args:(),
	args=(), 
	nprocs=2, 
	verbose=False
	):
	import sys, multiprocessing
	chunksize = int(N.ceil(len(data) / float(nprocs)))
	completed = multiprocessing.Value('i', 0)
	total = multiprocessing.Value('i', len(data))
	procs = []
	out_q = multiprocessing.Queue()
	def worker(oq, batch, num_completed, num_total, worker_id=-1):
		outdict = {}
		for i in range(0, len(batch)):
			chunk_id = chunksize * worker_id + i
			outdict[chunk_id] = func(batch[i], *args) 
			num_completed.value += 1
			sys.stdout.write(
				'\rworker ' +  str(worker_id) + \
				' (' + str(num_completed.value) + \
				' of ' + str(num_total.value) + ') '
				' computes RF # ' + str(chunk_id) + 10*' ' )
			sys.stdout.flush()
		if verbose: print '\r worker ' + str(worker_id) + ' has finished' + 30*' ' + '\r'
		oq.put(outdict)
	for i in range(nprocs):
		if verbose: print '\r worker ' + str(i) + ' started with chunk: ' + str(chunksize * i) + ' : ' + str(chunksize * (i + 1) )
		p = multiprocessing.Process(
			target=worker, 
			args=(out_q, 
				data[chunksize * i : chunksize * (i + 1)], 
				completed, 
				total, i))
		procs.append(p)
		p.start()
	resultdict = {}
	for p in procs:
		resultdict.update(out_q.get())
	for p in procs:
		p.join()
	return resultdict



'''-------------------------------------------------------------------------'''



from ..base.plots import write_rf_fit_debug_fig
def debug_write_fit(
	p, 
	f, 
	rf, 
	channel_w, 
	patch_w, 
	mode, 
	msg='foo',
	model='dog',
	path='.',
	scale=2.):
	rec = f(p, channel_w, patch_w, *N.indices(rf.shape))
	rec_err = (rf - rec)**2
	write_rf_fit_debug_fig(
		path + '/' + str(msg) + '_debug_' + str(model) + '_RF_fit.png', 
		rf, 
		patch_w, 
		mode, 
		p, 
		rec,
		rec_err, 
		model, 
		scale=scale)	



from ..base.receptivefield import arg_absmax_rfvector
def permutate_mu(
	n, 
	p, 
	patch_w,
	channel_w,
	mode,
	rf,
	ext_model=False
	):
	if n==0:
		center_point = N.array([patch_w/2., patch_w/2.])
	else:
		center_point = N.mean(arg_absmax_rfvector(mode, rf, patch_w, channel_w), axis=0)
	if n > 1:
		'''RF center point noise'''
		noise = (N.random.random_sample(2)-.5)*2 * patch_w/2#*.75
		center_point += noise
	p[0] = center_point[0]
	p[1] = center_point[1]
	return p



'''-------------------------------------------------------------------------'''



import cPickle
def pickle_fits(
	filepath, 
	result_dict,
	):
	_file = open(filepath, 'wb')
	cPickle.dump(result_dict, _file, -1)
	_file.close()
	print
	print 'pickled fits', filepath, 'written.'


def depickle_fits(
	filepath,
	suffix='fits',
	exit_on_errer=True
	):
	from ..base import match_filesuffix
	if match_filesuffix(filepath, suffix, verbose=False): pass
	else: 
		print 'A file with suffix \'.'+suffix+'\' is expected (filename.'+suffix+').'
		if exit_on_errer:
			print 'exiting.'
			exit()
		else:
			return None
	_file = open(filepath)
	result_dict = cPickle.load(_file)
	_file.close()
	return result_dict





