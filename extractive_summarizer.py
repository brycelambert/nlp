import sys, re
import nltk
import os
import string
import numpy
from collections import defaultdict
from IPython import embed
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from nltk import MaxentClassifier
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

#global store of a count vector and frequency matrix for tf_idf use
corpus_freq = (None, None)


#Topic class is intialized in loops through the data.
#It provides a 'document' or topic-wide tfidf lookup matrix
#to use when calculating a sentences individual feature set.
class Topic:
  def __init__(self, topic):
    self.subject = topic['topic']
    self.review_text = topic['lines']
    self.tfidf_table = self.build_tfidf_table()


  def features_for_sent(self, sent):
    return {
      'tf_idf_score': self.tf_idf_score(sent),
      'contains_subject_word': self.sentence_has_keyword(sent),
      'number_of_all_caps_words': self.number_of_all_caps_words(sent),
      'has_exlamation_point': self.has_exclamation_point(sent),
      'begins_with_capital_letter': self.begins_with_capital_letter(sent)
    }


  #private
  def tf_idf_score(self, sent):
    output = 0.0
    tokenized_sent = nltk.word_tokenize(sent.translate(None, string.punctuation))
    tf_idf_feat = corpus_freq[0].get_feature_names()

    for token in tokenized_sent:
      if token in tf_idf_feat:
        output += self.tfidf_table[tf_idf_feat.index(token.lower())]

    return output
  

  def sentence_has_keyword(self, sent):
    return int(self.subject not in sent)

  def number_of_all_caps_words(self, sent):
    return len(filter(lambda x: x.isupper(), sent.split()))

  def has_exclamation_point(self, sent):
    return int('!' in sent)

  def begins_with_capital_letter(self, sent):
    return int(sent[0].isupper())

  def build_tfidf_table(self):
    tfidf = TfidfTransformer(norm='l2')
    tfidf.fit(corpus_freq[1])
    review_freq = corpus_freq[0].transform(self.review_text)
    tfidf_matrix = tfidf.transform(review_freq)

    tfidf_matrix = tfidf_matrix.todense()
    tfidf_matrix = tfidf_matrix.tolist()[0]
    return tfidf_matrix

######## end

#RougeCalculator class is a container for score() ROUGE-2 score calculator
class RougeCalculator:

  def score(self, ext_summ, gold_summ):
    ext_summ_bigrams = self.gen_bigrams(ext_summ)
    gold_summ_bigrams = self.gen_bigrams(gold_summ)

    return \
    len(ext_summ_bigrams.intersection(gold_summ_bigrams)) / \
    len(gold_summ_bigrams)

  def gen_bigrams(self, s):
    s = ''.join(s).lower()
    tokens = RegexpTokenizer(r'\w+').tokenize(s)
    tokens = [t for t in tokens if not t in stopwords.words('english')]
    return set(ngrams(tokens, 2))

######## end

#Sets the corpus_freq global with with frequenct matrix and count vector for the corpus.
def set_corpus_frequencies(data):
  global corpus_freq

  text = [review['lines'] for review in data]
  text = [item for sublist in text for item in sublist]

  count = CountVectorizer()
  count = count.fit(text)
  matrix = count.transform(text)
  corpus_freq = (count, matrix)
  return True

def clean_lines(lines):
  output = []
  for line in lines:
    output.append( line.decode('utf-8', 'ignore').encode('utf8') )
  return output

#Reads in text from data set file in './data' and builds each topic
#into a dictionary with useful information. Helper function clean_lines()
#above handles some unicode parsing errors I encountered.
def parseData():
  output = []
  for topic_file in os.listdir('./data/topics'):
    if topic_file[0] is '.': continue
    lines = open('./data/topics/' + topic_file).readlines()
    lines = clean_lines(lines)
    name = topic_file.split('.')[0]
    topic_name = name.split('_')[0]
    topic_record = dict()
    topic_record['topic'] = topic_name
    topic_record['lines'] = lines
    topic_record['gold_std'] = \
      open('./data/summaries-gold/' + name + '/' + name + '.1.gold').readlines()
    output.append(topic_record)
  return output

#Loop through data to initialize an instance of Topic as needed, collect feature sets
#for topics and label them: 1 for a gold standard summary and 2 for otherwise
def get_training_feats(data):
  output = []

  for topic in data:
    t = Topic(topic)

    for summ_line in topic['gold_std']:
      output.append( (t.features_for_sent(summ_line), 1) )

    for review in topic['lines']:
      output.append( (t.features_for_sent(review), 0) )

  return output


def get_feats(data):
  output = []

  for topic in data:
    t = Topic(topic)

    for review in topic['lines']:
      output.append(t.features_for_sent(review))

  return output

#Accepts a topic and trained classifier.
#Returns the most probable 'summary quality' sentences
def extract_summary_for_topic(topic, classifier):
  t = Topic(topic)
  topic_feats = [t.features_for_sent(review) for review in topic['lines']]
  prob_classifications = classifier.prob_classify_many(topic_feats)

  # [pd.max() for pd in classified_test_data]
  # Classifer returns mostly zeroes, i.e sentences infrequently classifed as summary material
  # So let's just find the most likely indexes

  probs = [pd.prob(1) for pd in prob_classifications]
  sorted_indexes = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
  max_indexes = sorted_indexes[:len(topic['gold_std'])]

  # and use them to get our summary sentences
  return [topic['lines'][i] for i in max_indexes]


def main():
  #INTAKE DATA & BUILD TRAINING/TEST SETS
  reviews_corpus = parseData()
  set_corpus_frequencies(reviews_corpus)
  training_data = reviews_corpus[:26]
  test_data = reviews_corpus[26:]

  #BUILD MAXENT MODEL
  training_set = get_training_feats(training_data)
  classifier = MaxentClassifier.train(training_set)

  #CLASSIFY, EXTRACT & EVAL BY TOPIC
  scores = []
  baselines = []

  for topic in test_data:
    extracted_summary = extract_summary_for_topic(topic, classifier)
    random_summary = random.sample(topic['lines'], len(extracted_summary))

    score = RougeCalculator().score(extracted_summary, topic['gold_std'])
    baseline = RougeCalculator().score(random_summary, topic['gold_std'])

    scores.append(score)
    baselines.append(baseline)

    print "Summary for " + topic['topic'] + ':'
    print ''.join(extracted_summary)
    print "Rouge Score: " + str(score)


  print "Extracted Summary Rouge Average"
  print sum(scores) / len(scores)

  print "Baseline Summary Rouge Average"
  print sum(baselines) / len(baselines)



    
if __name__ == "__main__": 
  main()