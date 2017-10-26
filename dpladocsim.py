
from collections import OrderedDict
import ijson
import json
from json import JSONDecodeError
import os
import pandas as pd
import re
from scipy.spatial.distance import pdist
import xmltodict

import logging
logging.basicConfig(level=logging.DEBUG)

from nltk.corpus import stopwords

import gensim
from gensim import corpora, models, similarities


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

		if type(record) in [dict, OrderedDict]:
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

		## LDA
		# nltk stopwords
		self.stoplist = set(stopwords.words('english'))
		self.tokens = self.m_as_bow()


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


	def m_as_bow(self):
		
		'''
		get metadata as bag-of-words (bow)
		'''

		values = self.parse_m_values()
		words = " ".join(values)

		# save as tokens
		self.tokens = [word for word in words.lower().split() if word not in self.stoplist]
		return self.tokens


	def as_vec_bow(self, m):
		'''
		pass model (m) to doc, affix .doc_bow, using self.tokens
		'''
		self.vec_bow = m.id2word.doc2bow(self.tokens)
		return self.vec_bow


		

class RawRecord(object):

	'''
	Class to accomodate raw metadata
	'''

	def __init__(self, record_id, m_xml_string):

		# save id
		self.record_id = record_id

		# save metadata string
		self.m_xml_string = m_xml_string
		
		# convert to dictionary with xmltodict
		self.original_metadata = xmltodict.parse(self.m_xml_string)

		# alphnumeric characters used to calc vectors
		self.alnum = list('0123456789abcdefghijklmnopqrstuvwxyz')


	def parse_m_values(self):

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


	def m_as_char_vect_series(self):

		'''
		parse and count a-z,0-9, return as pandas series
		'''

		# grab characteres, stripping whitespace
		chars = ''.join(self.parse_m_values()).replace(' ','')
		
		# count
		char_counts = [ chars.count(char) for char in self.alnum ]

		# return as pandas series
		return pd.Series(char_counts, name=self.record_id, index=self.alnum)



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

		for i,r in enumerate(reader.dpla_record_generator()):
			self.add_record(r.m_as_char_vect_series())
			if i % 1000 == 0:
				logging.debug('loaded %s records' % i)


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

		return scores[:20]



class DocSimModelLDA(object):

	'''
	DocSim model using python Gensim LDA
	'''

	def __init__(self, reader=None, name=None):

		# get reader
		self.reader = reader

		self.texts = []
		self.article_hash = {}
		self.failed = []
		self.name = name


	def get_all_docs(self, limit=1000):

		count = 0
		for r in self.reader.dpla_record_generator():

			# get bow
			r.m_as_bow()

			self.texts.append(r.tokens)
			self.article_hash[r.dpla_id] = len(self.texts) - 1

			count += 1
			if count > limit:
				return 'limit reached'


	def gen_corpora(self):

		logging.debug("creating corpora dictionary for texts: %s.dict" % self.name)
		
		# creating gensim dictionary
		self.id2word = corpora.Dictionary(self.texts)
		self.id2word.save('%s/%s.dict' % ('models', self.name))

		# creating gensim corpus
		self.corpus = [self.id2word.doc2bow(text) for text in self.texts]
		'''
		Consider future options for alternate formats:
			Other formats include Joachim’s SVMlight format, Blei’s LDA-C format and GibbsLDA++ format.
			>>> corpora.SvmLightCorpus.serialize('/tmp/corpus.svmlight', corpus)
			>>> corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)
			>>> corpora.LowCorpus.serialize('/tmp/corpus.low', corpus)
		'''
		corpora.MmCorpus.serialize('%s/%s.mm' % ('models', self.name), self.corpus)

		logging.debug('finis.')


	def load_corpora(self):
		'''
		see above for selecting other corpora serializations
		this should also load dictionary
		'''
		# load corpora
		target_path = '%s/%s.mm' % ('models', self.name)
		if os.path.exists(target_path):
			logging.debug("loading serialized corpora: %s.mm" % self.name)
			self.corpus = corpora.MmCorpus(target_path)

		# load dictionary
		target_path = '%s/%s.dict' % ('models', self.name)
		if os.path.exists(target_path):
			logging.debug("loading serialized dictionary: %s.dict" % self.name)
			self.id2word = corpora.Dictionary.load(target_path)


	def gen_lda(self, num_topics=500, chunksize=100, passes=5):
		'''
		creates LDA model from mm corpora and dictionary
		'''
		self.lda = models.ldamulticore.LdaMulticore(
			corpus=self.corpus,
			id2word=self.id2word,
			num_topics=num_topics,
			chunksize=chunksize,
			passes=passes
		)
		self.lda.save('%s/%s.lda' % ('models', self.name))


	def load_lda(self):
		'''
		loads saved LDA model
		'''
		self.lda = models.ldamodel.LdaModel.load('%s/%s.lda' % ('models', self.name))


	def gen_similarity_index(self):
		self.index = gensim.similarities.MatrixSimilarity(self.lda[self.corpus])
		self.index.save('%s/%s.simindex' % ('models', self.name))


	def load_similarity_index(self):
		self.index = gensim.similarities.MatrixSimilarity.load('%s/%s.simindex' % ('models', self.name))


	def get_similar_records(self, input_record, limit=20):

		'''
		Run a couple of DPLARecord methods and query against model
		'''

		# get vectors of tokens against model
		input_record.as_vec_bow(self.lda)

		# query against model and get similarity matrix
		vec_lda = self.lda[input_record.vec_bow]
		input_record.sims = self.index[vec_lda]
		input_record.sims = sorted(enumerate(input_record.sims), key=lambda item: -item[1])

		# return by limit constraint
		return input_record.sims[:limit]


