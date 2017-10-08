

import ijson
import json
from json import JSONDecodeError
import pandas as pd
import re
from scipy.spatial.distance import pdist

'''
Using ijson to parse large json files:
https://pypi.python.org/pypi/ijson
'''


class Reader(object):


	def __init__(self, input_file):

		self.input_file = input_file
		self.file_handle = open(self.input_file)
		self.records_gen = ijson.items(self.file_handle, 'item')


	def get_next_dpla_record(self):

		return DPLARecord(next(self.records_gen))


	def dpla_record_generator(self):

		while True:
			yield self.get_next_dpla_record()



class ReaderRaw(object):


	def __init__(self, input_file):

		self.input_file = input_file
		self.file_handle = open(self.input_file,'rb')

		# bump file handle
		next(self.file_handle)
		self.records_gen = self.file_handle


	def get_next_dpla_record(self):

		r_string = next(self.file_handle).decode('utf-8').lstrip(',')
		return DPLARecord(r_string)


	def dpla_record_generator(self):

		while True:
			try:
				yield self.get_next_dpla_record()
			except JSONDecodeError:
				break



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
		self.dpla_id = self.record['_source']['id']
		self.dpla_url = self.record['_source']['@id']
		self.original_metadata = self.record['_source']['originalRecord']['metadata']
		self.metadata_string = str(self.original_metadata)
		self.char_counts = None

		# alphnumeric characters used to calc vectors
		self.alnum = list('0123456789abcdefghijklmnopqrstuvwxyz')



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

		vals = list(NestedDictValues(self.original_metadata))

		# remove empty vals
		return list(filter(None, vals))



	def m_as_set(self):

		'''
		Convert original metadata into comparable vectors
		'''

		return set(self.parse_m_values())


	def m_as_char_vect_series(self):

		'''
		parse and count a-z,0-9, return as pandas series
		'''

		# grab characteres, stripping whitespace
		chars = ''.join(self.parse_m_values()).replace(' ','')
		
		# count
		char_counts = [ chars.count(char) for char in self.alnum ]

		# return as pandas series
		return pd.Series(char_counts, name=self.dpla_id, index=self.alnum)

		

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



class DocSimModel(object):

	'''
	DocSimModel instance 

	Eventually, will want to calculate similarity of a newly introduced row

	See:
		https://en.wikipedia.org/wiki/Distance_matrix
		http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
		https://stackoverflow.com/questions/37003272/how-to-compute-jaccard-similarity-from-a-pandas-dataframe
		
		Particularly this:
			https://www.tensorflow.org/tutorials/wide

		Or, perhaps much simpler, would be euclidean distances:
			https://stackoverflow.com/a/39205919/1196358
	'''

	def __init__(self):

		# alphanumeric 
		# TODO: consider storing this somewhere more global
		self.alnum = list('0123456789abcdefghijklmnopqrstuvwxyz')

		# init empty dataframe
		self.df = pd.DataFrame(None, columns=self.alnum)


	def add_record(self, record_char_vect_series):

		self.df = self.df.append(record_char_vect_series)


	def train_model_from_reader(self, reader):

		for r in reader.dpla_record_generator():
			self.add_record(r.m_as_char_vect_series())


	def save_model(self, path):

		'''
		save model(df) to disk
		'''

		self.df.to_pickle(path)


	def load_model(self, path):

		'''
		load model from disk
		'''

		self.df = pd.read_pickle(path)


	def get_similar_records(self, input_record):

		'''
		Inefficient, but functional comparison of input record to model
		'''

		scores = []

		# loop through model
		for row in self.df.iterrows():
			scores.append((row[0], pdist([row[1], input_record], metric='euclidean')[0]))

		# sort and return top ten
		scores.sort(key=lambda tup: tup[1])

		return scores[:10]



