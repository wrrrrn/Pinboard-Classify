
import itertools
import analysis_tools
import get_data

data = []
corpus_file = '/_data/code/+projects/pinboard/vector_corpus.mm'
corpus_dct = analysis_tools.vector_corpus_dict_file
parser = analysis_tools.TextHandler()
pinboard = get_data.Pinboard("username","password") 
web = get_data.HtmlHandler()
vector_corpus = analysis_tools.Vector_Corpus()

def train():
	get_training_data("human")
	get_training_data("politics")
	get_training_data("philosophy")
	get_training_data("data")
	get_training_data("culture")

def get_training_data(category):
	print "\ngetting %s training data\n--------------------" % category
	links = pinboard.get_tag(category,count=75)
	for link in links:
		html = web.get_url(link['href'])
		if html:
			content = parser.parse_html(html)
			if content:
				print 'title: %s\nurl: %s\n' % (link['description'], link['href'])
				data.append(content)

def build_tfidf_vector_model(data):
	vector_corpus.build(data)
	analysis_tools.save_corpus(corpus_file,vector_corpus)
	# get tfidf model
	tfidf = analysis_tools.Tfidf_Model(vector_corpus,vector_corpus.corpus_dictionary)
	tfidf_corpus = tfidf.transform_corpus(vector_corpus)
	return tfidf

def load_tfidf_vector_model():
	vector_corpus = analysis_tools.load_corpus(corpus_file)
	vector_dict = analysis_tools.load_dict(corpus_dct)
	tfidf = analysis_tools.Tfidf_Model(vector_corpus,vector_dict)
	tfidf_corpus = tfidf.transform_corpus(vector_corpus)
	return tfidf

def classify(root_tag,count=15,save=True):
	links = pinboard.get_tag(root_tag,count)
	for link in links:
		incoming_content = web.get_url(link['href'])
		content = parser.parse_html(incoming_content)
		if content:
			tags = ['.classified']
			words = parser.get_words(content)
			tfidf_results = tfidf.classify(words,score=False)
			tags =  tags + analysis_tools.intersection(tfidf_results,pinboard.all_tags)
			print 'title: %s\nurl: %s' % (link['description'], link['href'])
			print '+suggested tags: %s' % (", ".join(tags))
			print '-tf-idf:%s\n' % (", ".join(tfidf_results))
			if save: pinboard.add(link['href'],link['description'],tags)

train()
#tfidf = build_tfidf_vector_model(data)
#tfidf = load_tfidf_vector_model()
classify(".classify",save=False)



