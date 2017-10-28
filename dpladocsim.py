
from collections import OrderedDict, Counter
import ijson
import json
from json import JSONDecodeError
import os
import pandas as pd
import re
from scipy.spatial.distance import pdist
import xmltodict

import logging
logging.basicConfig(level=logging.INFO)

from nltk.corpus import stopwords

import gensim
from gensim import corpora, models, similarities

# from elasticsearch import Elasticsearch
import elasticsearch as es
from elasticsearch_dsl import Search, A, Q



class Reader(object):


	def __init__(self, input_file):

		self.input_file = input_file
		self.file_handle = open(self.input_file)
		self.records_gen = ijson.items(self.file_handle, 'item')


	def get_next_dpla_record(self):

		return DPLARecord(next(self.records_gen))


	def dpla_record_generator(self, limit=False):

		i = 0
		while True:
			i += 1
			yield self.get_next_dpla_record()
			if limit and i >= limit:
					break



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


	def dpla_record_generator(self, limit=False, attr=None):

		i = 0
		while True:
			i += 1
			try:
				# if attr provided, return attribute of record
				if attr:
					yield getattr(self.get_next_dpla_record(), attr)
				# else, return whole record
				else:
					yield self.get_next_dpla_record()
				if limit and i >= limit:
					break
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


	def retrieve_docs(self, limit=1000):

		count = 0
		for r in self.reader.dpla_record_generator():

			# get bow
			r.m_as_bow()

			self.texts.append(r.tokens)
			self.article_hash[r.dpla_id] = len(self.texts) - 1

			# report every 1000
			if count % 1000 == 0:
				logging.debug('retrieving documents @ %s' % count)

			count += 1
			if count >= limit:
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


	def gen_lda(self, num_topics=500, chunksize=100, passes=5, multicore=False):
		
		'''
		creates LDA model from mm corpora and dictionary
		'''

		if multicore:
			logging.debug('multicore selected, using models.ldamulticore.LdaMulticore')
			self.lda = models.ldamulticore.LdaMulticore(
				corpus=self.corpus,
				id2word=self.id2word,
				num_topics=num_topics,
				chunksize=chunksize,
				passes=passes
			)

		else:
			logging.debug('not utilizing multicore, using models.ldamodel.LdaModel')
			self.lda = models.ldamodel.LdaModel(
				corpus=self.corpus,
				id2word=self.id2word,
				num_topics=num_topics,
				update_every=1,
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


	def get_similar_records(self, input_record, limit=20, parse_ambiguous=True):

		'''
		Run a couple of DPLARecord methods and query against model
		'''

		# get vectors of tokens against model
		input_record.as_vec_bow(self.lda)

		# query against model and get similarity matrix
		vec_lda = self.lda[input_record.vec_bow]
		input_record.sims = self.index[vec_lda]
		input_record.sims = sorted(enumerate(input_record.sims), key=lambda item: -item[1])

		# run ambiguity check if 2nd result is above 98%
		if parse_ambiguous:		
			if input_record.sims[1][1] > 0.99:
				logging.debug('running ambiguity check')
				input_record.sims = self.ambiguity_check(input_record)

		# return by limit constraint
		return input_record.sims[:limit]


	def ambiguity_check(self, input_record, iterations=25):

		'''
		run similarity test numerous times, accept most common answer
		'''

		checks = []
		for x in range(0, iterations):
			checks.append( self.get_similar_records(input_record, parse_ambiguous=False)[0] )

		# determine most common
		counts = [ (k, (v/100)) for k,v in Counter(elem[0] for elem in checks).items() ]
		counts = sorted(counts, key=lambda item: -item[1])

		return counts



class DocSimModelES(object):

	'''
	DocSim model using ElasticSearch index
	'''

	def __init__(self, reader=None, name=None, es_host='localhost'):

		self.reader = reader
		self.name = name
		self.es_handle = es.Elasticsearch(hosts=[es_host])

		self.indexing_failures = []


	def index_all_docs_to_es(self, limit=False):

		'''
		use reader, index all docs

		TODO: do in bulk, just testing here
		'''

		# create index
		try:
			self.es_handle.indices.delete(index=self.name, ignore=400)
		except:
			logging.info('could not delete index %s' % self.name)
		self.es_handle.indices.create(index=self.name, ignore=400)

		# iterate through rows ()		
		for i, r in enumerate(self.reader.dpla_record_generator(limit=limit)):

			# clean and prepare record
			body = r.record['_source']
			for val in ['_id', '_rev']:
				body.pop(val, None)

			# index
			try:
				self.es_handle.index(index=self.name, doc_type='record', id=r.dpla_id, body=body)
			except:
				logging.debug('could not index %s, reader index %s' % (r.dpla_id, i))

			if i % 500 == 0:
				logging.debug('indexed %s records' % i)


	def bulk_index_all_docs_to_es(self, limit=False):

		'''
		use streaming bulk indexer:
		http://elasticsearch-py.readthedocs.io/en/master/helpers.html
		'''

		def es_doc_generator(dpla_record_generator):

			for r in dpla_record_generator:
				for f in ['_id','_rev']:
					r['_source'].pop(f)
				yield r

		# index using streaming
		for i in es.helpers.streaming_bulk(self.es_handle, es_doc_generator(self.reader.dpla_record_generator(limit=limit, attr='record')), chunk_size=500):

			logging.info(i)


	def get_similar_records(self, input_record, limit=20):

		# instantiate search
		s = Search(using=self.es_handle, index=self.name)

		# build query and execute
		s = s.query('match', _all=' '.join(input_record.tokens))
		res = s.execute()

		# return
		scores = [ (hit.id,hit.meta.score) for hit in res ]
		return scores










