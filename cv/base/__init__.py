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

'''
base __init__.py
'''
import sys


def check_pid(pid):
	import os
	""" Check For the existence of a unix pid. """
	try:
		os.kill(pid, 0)
	except OSError:
		return False
	else:
		return True


def sum_sizeof(
	x
	):
	size = sys.getsizeof(x)
	if hasattr(x, '__iter__'):
		if hasattr(x, 'items'):
			for xx in x.items():
				size += sum_sizeof(xx)
		else:
			for xx in x:
				size += sum_sizeof(xx)
	return size


def show_sizeof(
	x, 
	level=0
	):
	print "\t" * level, x.__class__, sys.getsizeof(x), x
	if hasattr(x, '__iter__'):
		if hasattr(x, 'items'):
			for xx in x.items():
				show_sizeof(xx, level + 1)
		else:
			for xx in x:
				show_sizeof(xx, level + 1)


def dbgprintarr(
	x, 
	info='', 
	addInfo=False
	):
	import pprint as pp
	t = None
	if hasattr(x, 'shape'): t = x.shape
	print info, t, "\n", pp.pprint(x)
	if addInfo and t is not None: print "mean, ", x.mean(), " var, ", x.var()



def extract_filename_without_suffix(
	s
	):
	s = s.split('/')[-1] # TODO wont work on non unices! :O
	st = s.replace('../','').\
		replace('./','').\
		replace('.matr','').\
		replace('.rgc','').\
		replace('.map','').\
		replace('.fits','').\
		replace('.cov','').\
		replace('.coverage','').\
		replace('.py','').\
		replace('.png','').\
		replace('.jpg','').\
		replace('.tif','')
	return st



def match_filesuffix(
	s, 
	suffix='fits', 
	verbose=True
	):
	import re
	match = re.match(r'.*\.'+suffix+'$', s)
	if match is None:
		if verbose: print 'A file with pickled data and suffix \''+suffix+\
			  			  '\' is expected.\n(filename.'+suffix+').'
		return False
	else:
		return True	
		   


def print_elapsed(
	elapsed
	):
	from datetime import datetime, timedelta
	sec = timedelta(seconds=elapsed)
	d = datetime(1,1,1) + sec
	outstr = ''
	if d.day > 1:		outstr += str(d.day-1) + ' d '
	if d.hour > 0:		outstr += str(d.hour) + ' h '
	if d.minute > 0:	outstr += str(d.minute) + ' m '
	if d.second > 0:	outstr += str(d.second) + ' s'
	return outstr

	
def num_channel_of(mode):
	if   mode == 'lum': 					return 1
	elif mode == 'rg' or mode == 'rg_vs_b':	return 2
	elif mode == 'rgb':						return 3

def mode_of(num_channel):
	if num_channel == 1:
		return 'lum'
	elif num_channel == 2:
		return 'rg_vs_b' # oO
	elif num_channel == 3:
		return 'rgb'
		

def rgc_filename_str(
	**kwargs
	):
	if kwargs['clip']: 	strclip = '_clip'
	else: 	 			strclip = ''
	strnorm = 'p' + str(kwargs['p'])
	try:
		lr = kwargs['lr']
	except KeyError:
		lr = kwargs['lr1']
	return 'RGC(' + str(kwargs['vis']) + 'x' + str(kwargs['hid']) + ')' + \
		   '_' + kwargs['mode'] + '_' + '_k' + str(kwargs['k']) + \
		   '_' + strnorm + '_lr' + str(lr) + strclip


def v1_filename_str(
	**kwargs
	):
	try:				lr = kwargs['lr']
	except KeyError:	lr = kwargs['lr2']
	try:				hid = kwargs['hid']
	except:				hid = kwargs['hid1']
	return 'V1(' + str(hid) + 'x' + str(kwargs['hid2']) + ')' + \
		   '_' + kwargs['mode'] + '_' + \
		   '_a' + str(kwargs['a']) + \
		   '_b' + str(kwargs['b']) + \
		   '_d' + str(kwargs['d']) + \
		   '_lr' + str(lr)



def pad_sequence_num(
	num,
	padlen=8
	):
	strn = str(num)
	n_to_pad = padlen - len(strn)
	return '0'*n_to_pad+strn



def writer_directory_str(
	**kwargs	
	):
	from datetime import date
	datestr = str(date.today())
	if kwargs['which']=='rgc' or kwargs['which']=='drgc':
		misc_str = '_clip' if kwargs['clip'] else ''
		dim_str = str(kwargs['vis'])+'x'+\
				  str(kwargs['hid'])
	else:
		try:	hid = kwargs['hid']
		except:	hid = kwargs['hid1']
		misc_str = ''
		dim_str = str(kwargs['vis'])+'x'+\
				  str(hid)+'x'+\
				  str(kwargs['hid2'])
	writer_dir = str(kwargs['which'])+\
				 '_'+str(kwargs['mode'])+\
				 misc_str+'('+dim_str+')'\
				 +'_'+datestr
	import os
	return os.path.join(kwargs['odir'], writer_dir)



def incr_name_if_exists(s, n_tries, suffix=''):
	import os
	if n_tries == 0:
		slocal = s
	else:
		slocal = s+'-'+str(n_tries)
	if not os.path.exists(slocal+suffix):
		pass
	else:
		n_tries += 1
		slocal = incr_name_if_exists(s, n_tries, suffix=suffix)
	return slocal



def make_working_dir(
	**kwargs
	):
	def incr_dir_if_exists(s, n_tries):
		import os
		if n_tries == 0:
			slocal = s
		else:
			slocal = s+'-'+str(n_tries)
		if not os.path.exists(slocal):
			os.makedirs(slocal)
		else:
			n_tries += 1
			slocal = incr_dir_if_exists(s, n_tries)
		return slocal
	writer_dir = writer_directory_str(**kwargs)
	return incr_dir_if_exists(writer_dir, 0)



def remove_dir_and_nested_files(
	the_dir
	):
	import os
	if os.path.exists(the_dir):
		for the_file in os.listdir(the_dir):
			file_path = os.path.join(the_dir, the_file)
			try:
				if os.path.isfile(file_path):
					os.unlink(file_path)
			except Exception, e:
				print e
		os.removedirs(the_dir)



def create_dir(
	the_dir,
	remove_if_exists=False,
	):
	import os
	if remove_if_exists:
		remove_dir_and_nested_files(the_dir)
	if not os.path.exists(the_dir):	
		os.makedirs(the_dir)



def make_working_dir_sub(odir, name):
	import os
	sub_work_dir = os.path.join(odir,name)
	create_dir(sub_work_dir)
	return sub_work_dir



def write_param_info_file(
	the_dir,
	**kwargs
	):
	kwargs.pop('odir', None)
	kwargs.pop('tdir', None)
	kwargs.pop('map_outstr', None)
	kwargs.pop('exp_f', None)
	kwargs.pop('expcolor_f', None)
	kwargs.pop('writer_func', None)
	kwargs.pop('writer_logfile', None)
	kwargs.pop('which', None)
	import os
	f = open(os.path.join(the_dir,'train.params'), 'w')
	f.write(str(kwargs)+'\n')
	f.close()			



def make_filename(
	map_file,
	propstr,
	suffix,
	odir=None
	):
	import os
	if odir:
		working_dir = odir
	else:
		working_dir = os.path.split(map_file)[0]
	filename = os.path.join(working_dir, propstr)
	return incr_name_if_exists(filename, 0, suffix=suffix)











