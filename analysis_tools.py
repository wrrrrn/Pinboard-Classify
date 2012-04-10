from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from gensim import corpora, models, similarities
from decruft import Document
#from readability.readability import Document
import random
import pinboard
import nltk
import re 

vector_corpus_dict_file = '/_data/code/+projects/pinboard/vector_corpus.dict'

def intersection(list1,list2):
	set1 = set(list1)
	set2 = set(list2)
	return list(set1.intersection(set2))

def save_corpus(path,corpus):
		corpora.MmCorpus.serialize(path, corpus)

def load_corpus(path):
		return corpora.MmCorpus(path)

def load_dict(path):
		return corpora.Dictionary.load(path)

class TextHandler:
	def __init__(self):
		self.text = ""
		self.parsed_text = ""
		self.stopwords = stopwords.words('english')
		self.bigram_score_funcion = BigramAssocMeasures.chi_sq
		self.trigram_score_funcion = TrigramAssocMeasures.chi_sq
		self.top_ngram_count = 400

	def get_words(self,text,with_punctionation=True,remove_stopwords=False):
		self.text = text
		if with_punctionation:
			tokens = nltk.word_tokenize(self.text)
		else:
			words = nltk.word_tokenize(self.text)
			tokens = [w.lower() for w in words if re.match("[a-z\-\' \n\t]", w)]
			
		if remove_stopwords == True:
			tokens = [w.lower() for w in tokens if w not in self.stopwords]
		return tokens

	def get_sentences(self,text):
		self.text = text
		return nltk.sent_tokenize(self.text) 

	def get_bigrams(self,words):
		bigram_finder = BigramCollocationFinder.from_words(words)
		self.biagrams = bigram_finder.nbest(self.bigram_score_funcion , self.top_ngram_count)
		return self.biagrams 

	def get_trigrams(self,words):
		trigram_finder = TrigramCollocationFinder.from_words(words)
		self.trigrams = trigram_finder.nbest(self.trigram_score_funcion , self.top_ngram_count)
		return self.trigrams 

	def get_frequent_words(self,words,count=2000):
		all_words = nltk.FreqDist(w.lower() for w in words if re.match("[a-z\-\' \n\t]", w) and w not in self.stopwords)
		return all_words.keys()[:count]

	def summarize(self,text,keywords):
		i = 0
		found = []
		summary = []
		topics = keywords[:10]
		sentences = self.get_sentences(text)
		for sentence in sentences:
			words = self.get_words(sentence)
			if len(intersection(words,topics)) > 1:
				found.append(sentences[i-1])
				found.append(sentences[i])
				found.append(sentences[i+1])

		print len(found)
		"""if len(found) > 0 and len(found)/3 > 1:
			summ = found[:6]
		elif len(found) > 0 and len(found) < 3:
			summ = found
		else:
			summ = found"""
		summary = found
		summary =  ". ".join([s for s in summary])
		return summary.replace("\n", " ")

	def parse_html(self,raw):
		try:
			self.parsed_text = Document(raw).summary()
			try:
				self.parsed_text = nltk.clean_html(self.parsed_text)
			except Exception,e:
				print '->nltk.clean_html unable to parse raw html. Error: %s\n\
						-->return readability\'s results' % e
		except Exception,e:
			print '->Unable to parse raw html. Error: %s' % e
			return 0
		return self.parsed_text


class Bayes:
	def __init__(self):
		self.training_features = []
		self.test_features = []
		self.errors = []

	def set_word_features(self,features):
		self.word_features = features

	def set_feature_extractor(self,feature_extractor):
		self.feature_extractor = feature_extractor

	def build_classifier(self, content_category_data):
		content_category_data = self.randomize_content(content_category_data)
		print '\nBuilding Bayes Classification Features'
		category_class_features = [(self.feature_extractor(content_category[0]),content_category[1]) \
									for content_category in content_category_data]

		category_class_cutoff = len(category_class_features)*3/4

		self.training_features = category_class_features[:category_class_cutoff] 
		self.test_features = category_class_features[category_class_cutoff:] 
		print 'train on %d instances, test on %d instances' % (len(self.training_features), len(self.test_features))

		self.classifier = NaiveBayesClassifier.train(self.training_features)

	def test_classifier(self):
		self.classifier.show_most_informative_features(15)
		print 'accuracy:',nltk.classify.accuracy(self.classifier, self.test_features),'\n'

	def classify(self,content):
		self.classification = self.classifier.classify(self.feature_extractor(content))
		return self.classification

	def randomize_content(self,content_list):
		new_list = []
		while content_list:                        
		    element = random.choice(content_list)
		    new_list.append(element)
		    content_list.remove(element)
		return new_list

class Vector_Corpus(object):
	def __init__(self,build=True):
		self.text_parser = TextHandler()
		self.corpus_dictionary_words = []
		self.corpus_content = []
		if not build:
			 self.corpus_dictionary_words =m
		
	def __iter__(self):
		for doc in self.corpus_content:
			# assume there's one document per line, tokens separated by whitespace
			yield self.corpus_dictionary.doc2bow(doc)

	def create_corpus(self,content):
		for doc in content:
			document_words = self.text_parser.get_words(doc,remove_stopwords=True,with_punctionation=False)
			self.corpus_content.append(document_words)

	def build(self,content):
		self.content = [c for c in content if isinstance(c,basestring)]
		for doc in self.content:
			document_words = self.text_parser.get_words(doc,remove_stopwords=True,with_punctionation=False)
			all_tokens = sum([document_words], [])
			tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
			stripped_document_words = [word for word in document_words if word not in tokens_once]
			self.corpus_dictionary_words.append(stripped_document_words)
		self.corpus_dictionary = corpora.Dictionary(self.corpus_dictionary_words)
		self.corpus_dictionary.save(vector_corpus_dict_file)
		self.create_corpus(self.content)

class Tfidf_Model(object):
	def __init__(self,corpus,dictionary):
		self.dictionary_map = {}
		self.text_parser = TextHandler()
		self.dictionary = dictionary
		self.dictionary_mapping()
		print '\nGenerating TF-IDF Model'
		self.tfidf = models.TfidfModel(corpus)

	def dictionary_mapping(self):
		token2id = self.dictionary.token2id
		for word, word_id in token2id.items():
			self.dictionary_map[word_id] = word

	def transform(self,words):
		#words = self.text_parser.get_words(words,remove_stopwords=True,with_punctionation=False)
		return self.tfidf[self.dictionary.doc2bow(words)]

	def transform_corpus(self,corpus):
		return self.tfidf[corpus]

	def classify(self,words,score=False):
		doc = self.transform(words)
		tf = sorted(
				[(self.dictionary_map[word_score[0]],word_score[1]) for word_score in doc]
				,key=lambda word: word[1])
		if score:
			return tf
		else:
			tf_id2word = [] 
			for word_score in tf:
				try: 
					mapped_word = self.dictionary_map[word_score[0]]
					tf_id2word.append(mapped_word)
				except:
					tf_id2word.append(word_score[0])
			top_tf_words = tf_id2word[-20:]
			return sorted(top_tf_words, reverse=True)

