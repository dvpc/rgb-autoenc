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


def fit_map(args):
	if args.dbg:
		from parametric_fit import fit_debug as fitmap
		if args.n:
			args.begin = args.n
			args.num_iter = 1
			args.ncpu = 1
	else:		
		from parametric_fit import fit_map_pickle_result as fitmap
	fitmap(args.file,
		   args.begin,
		   args.num_iter,
		   args.model,
		   args.nfits,
		   args.maxitr,
		   args.ncpu,
		   args.odir)


def rec_map(args):
	from parametric_fit import rec_map_from_pickled_result
	rec_map_from_pickled_result(
		args.file, 
		args.odir)


def cluster_fits(args):
	from cluster import cluster_map
	return cluster_map(args)




