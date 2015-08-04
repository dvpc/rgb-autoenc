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

import argparse


parser = argparse.ArgumentParser(prog='analyze')
parser.add_argument('file', type=str, help='input file')
parser.add_argument('-odir', type=str, help='output directory', default=None)
parser.add_argument('-odirnosub', help='dont create subdir in odir. Only considered by proto yet.', action='store_true')

subparsers = parser.add_subparsers()


'''parametric fits'''
parser_a = subparsers.add_parser('fit', help='fit map and pickle result')
parser_a.set_defaults(which='fit')

parser_a.add_argument('-dbg', help='debug mode no pickled result written but lots of files.', action='store_true')
parser_a.add_argument('-n', type=int, help='shortcut for b = n and i = 1. ignored in non-debug.')
parser_a.add_argument('-b' ,'--begin', type=int, help='begin fitting with RF number b')
parser_a.add_argument('-i' ,'--num_iter', type=int, help='num iterations')

parser_a.add_argument('-model', type=str, default='edog', choices=['dog', 'edog'])
parser_a.add_argument('-maxitr', type=int, help='max num of solver iterations', default=5000)
parser_a.add_argument('-nfits', type=int, help='num of best fit attempts', default=15)
parser_a.add_argument('-ncpu', type=int, help='num of cpu', default=4)


'''reconstruction of parametric fits'''
parser_b = subparsers.add_parser('rec', help='reconstruct map of fit results')
parser_b.set_defaults(which='rec')


'''clustering of parametric fit data'''
parser_c = subparsers.add_parser('cl', help='cluster fit results')
parser_c.set_defaults(which='cl')
parser_c.add_argument('-n', type=int, help='num cluster', default=2)
parser_c.add_argument('-t', type=float, help='threshold', default=.155)
parser_c.add_argument('-alg', type=str, help='cluster algo', choices=['kmean', 'spec', 'hflat', 'hcomp', 'hweight', 'havg'], default='kmean')

parser_c.add_argument('-fold', help='fold pairwise opponent clusters into one cluster.', action='store_true')
parser_c.add_argument('-rec', help='use color of reconstructed map', action='store_true')
parser_c.add_argument('-csp', type=str, help='colorspace', choices=['rgb', 'yiq'], default='rgb')

parser_c.add_argument('-nz', type=float, help='nonzero', default=0)
parser_c.add_argument('-chr', type=float, help='chromaticy', default=0)
parser_c.add_argument('-err', type=float, help='reconstruction error', default=0)
parser_c.add_argument('-surr', help='use surround color approx', action='store_true')

parser_c.add_argument('-plot', help='plot cluster fits coverage', action='store_true')
parser_c.add_argument('-pr', help='prune map', action='store_true')

'''plot cluster cov'''
parser_f = subparsers.add_parser('clp', help='plot cluster fit results. coverage')
parser_f.set_defaults(which='clp')
parser_f.add_argument('-scale', type=float, help='scaling of all ellipsoids', default=2)
parser_f.add_argument('-pad', type=int, help='padding (between suplots) in inches', default=.1)
parser_f.add_argument('-dpi', type=int, help='resolution of plot (dots per inch)', default=288)
parser_f.add_argument('-a', type=float, help='alpha', default=1.)
parser_f.add_argument('-plain', help='only plot RF fits, no metrics...', action='store_true')

parser_g = subparsers.add_parser('clsp', help='plot cluster fit results. colorspace')
parser_g.set_defaults(which='clsp')
parser_g.add_argument('-pad', type=int, help='padding (between suplots) in inches', default=.1)
parser_g.add_argument('-dpi', type=int, help='resolution of plot (dots per inch)', default=288)
parser_g.add_argument('-a', type=float, help='alpha', default=1.)
parser_g.add_argument('-plain', help='only plot RF fits, no metrics...', action='store_true')
parser_g.add_argument('-single', help='only plot 3d', action='store_true')

parser_h = subparsers.add_parser('clprint', help='print clustered data')
parser_h.set_defaults(which='clprint')

parser_j = subparsers.add_parser('sortcl', help='sort clusters by giving order manually.')
parser_j.set_defaults(which='sortcl')
parser_j.add_argument('ids', metavar='N', type=int, nargs='+', help='list of new order of clusters')

'''prune map'''
parser_d = subparsers.add_parser('pr', help='prune map')
parser_d.set_defaults(which='pr')
parser_d.add_argument('-scol', help='sort pruned map RF column-wise, (default row-wise)', action='store_true')
parser_d.add_argument('-norm', help='normalize RF weights oof whole map', action='store_true')

'''debug pickled fits'''
parser_e = subparsers.add_parser('print', help='depickle debug print to console')
parser_e.set_defaults(which='print')
parser_e.add_argument('-all', help='if True print everything', action='store_true')


'''make proto filters'''
parser_d = subparsers.add_parser('proto', help='make proto-filters for convoling')
parser_d.set_defaults(which='proto')
parser_d.add_argument('-model', type=str, choices=['dog', 'edog', ], default=None)
parser_d.add_argument('-mean', help='if true: use mean reconstruction, if false, chose prototype rf from map (emulate handchosen)', action='store_true')
parser_d.add_argument('-mlen', type=int, help='mean: include number of fits per channel in mean reconstruction', default=None)
parser_d.add_argument('-synpos', help='mean: use synthetic position of mean filters', action='store_true')
parser_d.add_argument('-maxd', type=float, help='not mean: include RF in max dist to vis center', default=3)
parser_d.add_argument('-dist', help='not mean: if True, select prototype filter by distance to vis center', action='store_true')
parser_d.add_argument('-thr', type=float, help='not mean: threshold of abs max weight. RF below are not considered', default=0.7)
parser_d.add_argument('-ids', metavar='N', type=int, nargs='+', help='not mean: list of RF indices, one for each channel. if unspecified, prototypes are chosen automatically.')
parser_d.add_argument('-dbg', help='graph additional info on prototype RF', action='store_true')




'''convole'''
parser_e = subparsers.add_parser('conv', help='convole')
parser_e.set_defaults(which='conv')
parser_e.add_argument('-image', type=str, help='image file', default=None)
parser_e.add_argument('-lum', help='treat kernel 0 and 1 as White and Black', action='store_true')
parser_e.add_argument('-olum', help='output is converted (if RGB) to luminosity (r+g+b)', action='store_true')


'''spatal histogram of RF weights'''
parser_i = subparsers.add_parser('hist', help='make spatal histogram of RF weights')
parser_i.set_defaults(which='hist')
parser_i.add_argument('-thr', type=float, help='threshold of abs max weight. RF below are not considered', default=0.7)


'''histogram of RF size'''
parser_j = subparsers.add_parser('sh', help='make histogram of RF sizes')
parser_j.set_defaults(which='sh')
parser_j.add_argument('-lum', help='consider distinct luminosity channels. Default False', action='store_true')


'''color distribution of input data set'''
parser_j = subparsers.add_parser('cd', help='estimate color distribution of input data set')
parser_j.set_defaults(which='cd')
parser_j.add_argument('-rgb', help='plot RGB, default HSV', action='store_true')
parser_j.add_argument('-all', help='true: whole image, false: only a patch', action='store_true')
parser_j.add_argument('-n', type=int, help='num patches', default=100)
parser_j.add_argument('-skew', type=int, help='skew intensity by deg', default=30)

args = parser.parse_args()


'''set out directory if unspecified, to the same directory of the file'''
if args.odir == None:
	import os
	args.odir = os.path.split(args.file)[0]



import time
start_time = time.time()


if args.which == 'fit':
	from cv.analyze import fit_map
	fit_map(args)

elif args.which == 'rec':
	from cv.analyze import rec_map
	rec_map(args)

elif args.which	== 'cl':
	from cv.analyze import cluster_fits
	cluster_fits(args)

elif args.which == 'clp':
	from cv.analyze.cluster import plot_cluster_cov
	plot_cluster_cov(args)

elif args.which	== 'clsp':
	from cv.analyze.cluster import plot_clusters_colorspace
	plot_clusters_colorspace(args)

elif args.which == 'clprint':
	from cv.analyze.cluster import print_cluster_data
	print_cluster_data(args)

elif args.which == 'sortcl':
	from cv.analyze.sortcl import sort_cl_fits
	sort_cl_fits(args.file, args.odir, args.ids)

elif args.which == 'pr':
	from cv.analyze.prune import prune_map
	prune_map(args.file, args.odir, not args.scol, args.norm)

elif args.which == 'print':
	from cv.analyze.parametric_fit import print_pickled_fits
	print_pickled_fits(args.file, args.all)

elif args.which == 'proto':
	from cv.analyze.proto_filter import make_proto_filter
	make_proto_filter(args.file, args.odir, args.model, 
		synthetic_pos=args.synpos, 
		mean_rec=args.mean,
		max_dist_to_center=args.maxd,
		meanlen=args.mlen,
		min_err_metric=not args.dist,
		abs_weight_threshold=args.thr,
		indices=args.ids,
		debug=args.dbg,
		odir_nosubdir=args.odirnosub)

elif args.which == 'conv':
	from cv.analyze.convole import convole_image_with_filter
	convole_image_with_filter(
		args_file=args.file, 
		input_image=args.image, 
		odir=args.odir,
		lum_channels=args.lum,
		odir_lum_channels=args.olum,
		odir_nosub=args.odirnosub)

elif args.which == 'hist':
	from cv.analyze.spat_hist import make_spatal_hist
	make_spatal_hist(args.file, args.odir, args.thr)

elif args.which == 'sh':
	from cv.analyze.size_hist import make_size_hist
	make_size_hist(args.file, args.odir, args.odirnosub, args.lum)

elif args.which == 'cd':
	if args.rgb:
		from cv.analyze.colordist import est_color_distr_rgb as est_color
	else:
		from cv.analyze.colordist import est_color_distr as est_color
	est_color(args_file=args.file, args_odir=args.odir, 
		args_n=args.n, args_all=args.all, args_skew=args.skew)

elapsed = time.time() - start_time
from cv.base import print_elapsed
print 'elpased ', print_elapsed(elapsed)

