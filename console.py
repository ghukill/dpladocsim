# console utility for pydocsim

import re
import time

from dpladocsim import *

# DEBUG AUTO LOAD
reader = Reader('data/michigan.json')
r1 = reader.get_next_dpla_record()
r2 = reader.get_next_dpla_record()
r3 = reader.get_next_dpla_record()
r4 = reader.get_next_dpla_record()


def get_new_reader():
	return Reader('data/michigan.json')


def get_new_raw_reader():
	return ReaderRaw('data/michigan.json')


def benchmark_raw_read():
	raw_reader = get_new_raw_reader()
	count = 0
	stime = time.time()
	for r in raw_reader.dpla_record_generator():
		count += 1
	print("count %s, time elapsed %s" % (count, (time.time()-stime)))


def benchmark_ijson_read():
	ijson_reader = get_new_reader()
	count = 0
	stime = time.time()
	for r in ijson_reader.dpla_record_generator():
		count += 1
	print("count %s, time elapsed %s" % (count, (time.time()-stime)))
