import pinboard
import urllib2
import StringIO
import gzip


class Pinboard():
	def __init__(self,username,pwd):
		self.all_tags = []
		self.pinboard_account = pinboard.open(username,pwd)
		self.get_tags()

	def get_tag(self,idea,count=10):
		self.pinboard_links = self.pinboard_account.posts(tag=idea,count=count)
		print 'got links'
		#self.urls = [link['href'] for link in self.pinboard_links]
		return self.pinboard_links

	def get_tags(self):
		tags = self.pinboard_account.tags()
		for tag in tags:
				self.all_tags.append(tag['name'])
		return self.all_tags 

	def add(self,url,description,tags,extended=""):
		self.pinboard_account.add(url,description,extended,tags)

class HtmlHandler():
	def __init__(self):
		self.excluded_urls = \
		['http://www.philosophyofinformation.net/publications/pdf/htdpi.pdf']

	def get_url(self,content_url):
			if content_url not in self.excluded_urls:
				#print "getting url: %s" % content_url
				self.html = self.get_webpage(content_url)
				return self.html	

	def get_webpage(self,address):
		request = urllib2.Request(address)
		request.add_header('Accept-encoding', 'gzip')
		try:
			response = urllib2.urlopen(request)
			if response.info().get('Content-Encoding') == 'gzip':
				data = StringIO.StringIO(response.read())
				gzipper = gzip.GzipFile(fileobj=data)
				html = gzipper.read()
			else:
				html = response.read()
		except Exception,e:
			print e
			return False
		return html

