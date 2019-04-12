
import argparse

parser = argparse.ArgumentParser(prog='playground')
parser.add_argument('-odir', type=str, default=None, help='output directory')
parser.add_argument('-mode', type=str, default='rgb', choices=['rgb', 'rg', 'rg_vs_b', 'lum'])
parser.add_argument('-model', type=str, default='edog_ext', choices=['dog', 'edog', 'edog_ext', 'edog_ext2'])

parser.add_argument('-pw', type=int, default=13, help='RF square patch size')
parser.add_argument('-scale', type=float, default=2, help='scaling of the ellipse plot')

parser.add_argument('params', metavar='P', type=float, nargs='+', help='fit parameters')


args = parser.parse_args()

if args.odir == None:
	args.odir = '.'

print 
# print args
print 


'''
Num Params
			lum		rg	 	rgb
dog			6 		8 		10	 		
edog 		8 		10 		12
edog_ext 	- 		- 		17
edog_ext2 	-		-		16
'''
def __chk_sufficient_params(num_p, allowed=[6,8,10]):
	if args.mode == 'lum':
		return num_p - allowed[0]
	elif args.mode == 'rg' or args.mode == 'rg_vs_b':
		return num_p - allowed[1]
	elif args.mode == 'rgb':
		return num_p - allowed[2]
	else:
		return -1

def __select_rec_fun():
	if args.mode == 'lum':
		if args.model == 'dog':
			from cv.analyze.dog import dog_a as f
		elif args.model == 'edog':
			from cv.analyze.edog import edog_a as f
		else:
			f = None
	elif args.mode == 'rg' or args.mode == 'rg_vs_b':
		if args.model == 'dog':
			from cv.analyze.dog import dog_ab as f
		elif args.model == 'edog':
			from cv.analyze.edog import edog_ab as f
		else:
			f = None
	elif args.mode == 'rgb':
		if args.model == 'dog':
			from cv.analyze.dog import dog_abc as f
		elif args.model == 'edog':
			from cv.analyze.edog import edog_abc as f
		elif args.model == 'edog_ext':
			from cv.analyze.edog import edog_abc_ext as f			
		else:
			from cv.analyze.edog import edog_abc_ext2 as f
	else:
		f = None
	return f


num_params = len(args.params)
if args.model == 'dog':
	dparam = __chk_sufficient_params(num_params, allowed=[7,9,11])
elif args.model == 'edog':
	dparam = __chk_sufficient_params(num_params, allowed=[9,11,13])
elif args.model == 'edog_ext':
	dparam = __chk_sufficient_params(num_params, allowed=[0,0,18])
elif args.model == 'edog_ext2':
	dparam = __chk_sufficient_params(num_params, allowed=[0,0,17])


assert dparam >= 0, str(abs(dparam)) + ' Parameters are Missing'


from cv.base import num_channel_of
n_channel = num_channel_of(args.mode)
channel_w = args.pw**2

import numpy as N
rf = N.zeros(args.pw**2*n_channel)

f = __select_rec_fun()
assert f != None, 'Model ' + str(args.model) + ' is not implemented for mode: '+str(args.mode)

rec = f(args.params, channel_w, args.pw, *N.indices(rf.shape))



from cv.analyze.util import debug_write_fit

debug_write_fit(args.params, f, rf, channel_w, args.pw, 
	args.mode, msg='foo', model=args.model, path=args.odir, scale=args.scale)






