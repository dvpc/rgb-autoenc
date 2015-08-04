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

import os

from datetime import datetime
from daemon import Daemon


class ColorDaemon(Daemon):

	def __init__(self):
		self.time_started = None

		self.daemon_dir = './var_run_colord'
		self.pid_file = self.daemon_dir+'/colord.pid'
		super(ColorDaemon, self).__init__(self.pid_file)
		self.deamon_log = self.daemon_dir+'/colord.log'
		self.child_pid_file = self.daemon_dir+'/colord_child.pid'

		self.__job_args = None
		self.children = []

	def __write_log(self, msg, mode='a'):
		f = open(self.deamon_log, mode)
		f.write(msg)
		f.close()

	def __get_pid(self):
		pid = None
		'''check if pid file exists'''
		if os.path.exists(self.pid_file):
			f = open(self.pid_file, 'r')
			pid = int(f.read())
			f.close()
		return pid		

	def __get_pid_of_running_deamon(self):
		pid = self.__get_pid()
		from cv.base import check_pid
		if pid and not check_pid(pid): 
			pid = None
		return pid

	def is_deamon_running(self, return_pid=False):
		'''check if pid file exists'''
		pid = self.__get_pid()
		'''check a process with the pid is running'''
		from cv.base import check_pid
		if return_pid:
			if pid:	return check_pid(int(pid)), pid
			else:	return False, -1
		else:
			if pid:	return check_pid(int(pid))
			else:	return False

	def start_daemon(self):
		'''create working dir if necessarry'''
		if not os.path.exists(self.daemon_dir):
			os.makedirs(self.daemon_dir)
		else:
			'''if not running remove pid_file'''
			if not self.is_deamon_running(self.pid_file):
				if os.path.exists(self.pid_file):
					os.remove(self.pid_file)
		'''start daemon'''
		self.start()

	def stop_deamon(self):
		import signal
		'''terminate all children'''
		if os.path.exists(self.child_pid_file):
			f = open(self.child_pid_file, 'r')
			child_str = f.read()
			f.close()		
			for p in child_str.split(','):
				if p: os.kill(int(p), signal.SIGKILL)
		if os.path.exists(self.child_pid_file):
			os.remove(self.child_pid_file)
		'''check if process with the pid is running'''
		pid = self.__get_pid_of_running_deamon()
		'''if running stop process'''
		if pid:	
			os.kill(pid, signal.SIGKILL)
		if os.path.exists(self.pid_file):
			os.remove(self.pid_file)

	def job_initialize(self, args):
		from cv.base import create_dir
		create_dir(args.tdir, remove_if_exists=True)
		self.__job_args = args

	def run(self):
		# self.__write_log('', 'w')
		self.__write_log('-------------------------------------------------------------------------------\n',)
		self.__write_log(datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S\n'))
		self.__write_log(str(self.__job_args)+'\n')
		self.__write_log('-------------------------------------------------------------------------------\n',)

		if self.__job_args:
			from cv.train import start_training_from_args
			'''note: need nonblocking spawn_writer_process=True here!'''
			childs = start_training_from_args(self.__job_args, 
				spawn_writer_process=True, writer_logfile=self.deamon_log)
			self.children = [p.pid for p in childs]

			'''write child pid file'''
			f = open(self.child_pid_file, 'w')
			child_str = ''
			for c in self.children:
				child_str += str(c)+','
			f.write(child_str)
			f.close()
		else:
			self.__write_log('no job specified. idling.', 'a')		











