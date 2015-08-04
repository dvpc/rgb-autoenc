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


parser = argparse.ArgumentParser(prog='train')

parser.add_argument('-daemon', help='if true: run as daemon', action='store_true')

parser.add_argument('-odir', type=str, help='output directory', default='./tmp')
parser.add_argument('-tdir', type=str, help='temp directory', default='./tmp')
parser.add_argument('-datadir', type=str, help='image data directory', default='../da/data/Foliage')
parser.add_argument('-dtype', type=str, help='image type', default='tif')

parser.add_argument('-maxep', type=int, help='max training epochs', default=32000000)
parser.add_argument('-outn', type=int, help='output for each n steps', default=1000)
parser.add_argument('-outseq', help='output sequence of images', action='store_true')

parser.add_argument('-ncpu', type=int, help='num of processes', default=1)
parser.add_argument('-npt', type=int, help='num patches per image', default=3)
parser.add_argument('-th', help='if true: use theano else: numpy only', action='store_true')
parser.add_argument('-ptp', help='if true: dont subtract the patch\'s mean from the image patch', action='store_true')
parser.add_argument('-swp', type=int, help='patch chance to sawp axes (0-100)', default=0)

parser.add_argument('-btchn', type=int, help='number of patches in a minibatch', default=512)

parser.add_argument('-mode', type=str, choices=['rgb', 'rg', 'rg_vs_b', 'lum'], default='lum')


subparsers = parser.add_subparsers()

parser_a = subparsers.add_parser('rgc', help='train rgc layer only')
parser_a.set_defaults(which='rgc')
parser_a.add_argument('-rgcfile', type=str, help='continue from .map file', default=None)
parser_a.add_argument('-lr', type=float, help='model parameter: lr', default=.0555)
parser_a.add_argument('-vis', type=int, help='model parameter: sqrt of visible units', default=8)
parser_a.add_argument('-hid', type=int, help='model parameter: sqrt of hidden units', default=8)
parser_a.add_argument('-clip', help='model parameter: clip hid weights (ReLU)', action='store_true')
parser_a.add_argument('-leaky', type=float, help='model parameter: leaky ReLU lower bound.', default=0)
parser_a.add_argument('-k', type=float, help='model parameter: k', default=1.1111e-5)
parser_a.add_argument('-p', type=float, help='model parameter: p', default=.5)


parser_b = subparsers.add_parser('v1', help='train v1 only with fixed rgc layer')
parser_b.set_defaults(which='v1')
parser_b.add_argument('rgcfile', type=str, help='fixed rgc map. sets -vis', default=None)
parser_b.add_argument('-v1file', type=str, help='continue from .map file', default=None)
parser_b.add_argument('-lr', type=float, help='model parameter: lr', default=.0555)
parser_b.add_argument('-hid2', type=int, help='model parameter: sqrt of hidden units', default=8)
parser_b.add_argument('-a', type=float, help='model parameter: tf sparseness', default=0.7)
parser_b.add_argument('-b', type=float, help='model parameter: tf scale', default=1.9)
parser_b.add_argument('-decay', type=float, help='model parameter: decay', default=.001)
parser_b.add_argument('-amp', type=float, help='model parameter: scale rgc output', default=.5)
parser_b.add_argument('-mom', type=float, help='momentum', default=.0)




args = parser.parse_args()

if args.daemon:
	from cv.daemon.colord import ColorDaemon
	app = ColorDaemon()
	is_running, pid = app.is_deamon_running(return_pid=True)
	if is_running:
		print 
		print 'colord is already running with pid ' + str(pid) + '.'
		print 'stop deamon first:    $python colord.py stop'
		print 
		exit()
	else:
		app.job_initialize(args)
		app.start()
else:
	from cv.train import start_training_from_args
	start_training_from_args(args)
	


