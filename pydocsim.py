

import numpy as np
import re

import ijson

'''
Using ijson to parse large json files:
https://pypi.python.org/pypi/ijson
'''


class Reader(object):


	def __init__(self, input_file):

		self.input_file = input_file
		self.file_handle = open(self.input_file)
		self.records_gen = ijson.items(self.file_handle, 'item')


	def get_next_metadata_record(self):

		'''
		returns tuple of 
		'''

		return DPLARecord(next(self.records_gen))


class DPLARecord(object):


	def __init__(self, record):

		'''
		expecting dictionary or json of record
		'''

		if type(record) == dict:
			self.record = record
		elif type(record) == str:
			self.record = json.loads(record)

		# capture convenience values
		self.pre_hash_dpla_id = self.record['_id']
		self.dpla_id = self.record['_source']['_id']
		self.dpla_url = self.record['_source']['@id']
		self.original_metadata = self.record['_source']['originalRecord']['metadata']
		self.metadata_string = str(self.original_metadata)


	def parse_m_values(self):

		'''
		attempt to retrieve only terminating values from nested dictionary of DPLA representation of original metadata

		Returns:
			list
		'''

		def NestedDictValues(i):

			if isinstance(i, dict):
				for v in i.values():
					if isinstance(v, dict):
						yield from NestedDictValues(v)
					elif isinstance(v, list):
						yield from NestedDictValues(v)
					else:
						yield v
					
			elif isinstance(i, list):
				for v in i:
					if isinstance(v, dict):
						yield from NestedDictValues(v)
					elif isinstance(v, list):
						yield from NestedDictValues(v)
					else:
						yield v

		return list(NestedDictValues(self.original_metadata))



	def m_as_set(self):

		'''
		Convert original metadata into comparable vectors
		'''

		return set(self.parse_m_values())
		





class RecordCompare(object):

	@staticmethod
	def compare(r1, r2):

		'''
		compare two DPLARecords
		'''
		r1_s = r1.m_as_set()
		r2_s = r2.m_as_set()
		overlap = r1_s & r2_s
		return float(len(overlap)) / float(sum([len(r1_s),len(r2_s)]) / 2)
