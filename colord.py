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

from cv.daemon.colord import ColorDaemon


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(prog='colord')
	parser.add_argument('op', type=str, choices=['start', 'stop', 'restart', 'status'])
	args = parser.parse_args()

	app = ColorDaemon()

	if args.op == 'start':
		app.start_daemon()
	elif args.op == 'stop':
		app.stop_deamon()
		print 'colord stopped.'
	elif args.op == 'restart':
		app.stop_deamon()
		print 'colord stopped.'
		app.start_daemon()
	elif args.op == 'status':
		is_run, pid = app.is_deamon_running(return_pid=True)
		if is_run:
			print 'colord is running with pid ' + str(pid)
		else:
			print 'colord is not running.'





