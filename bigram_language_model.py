import sys
from collections import defaultdict
from math import log, exp
from nltk.corpus import brown

from IPython import embed

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.

######################################################
def makeVocab(corpus):
  corpus_hash = defaultdict(int)

  for line in corpus:
    for word in line: corpus_hash[word] += 1

  vocab = set(
    (k) for k,v in corpus_hash.items() if v > 1
  )

  vocab.update({ unknown_token, start_token, end_token })
  return vocab

def PreprocessText(corpus, vocab):
  output = list(corpus)

  for line in output:
    line.insert(0, start_token)
    line.append(end_token)
    
    for i, word in enumerate(line):
      if word not in vocab: line[i] = unknown_token

  return output

########################################################


class BigramLM:
    def __init__(self, vocabulary, training_corpus, smoothing=False):
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.unigram_counts = defaultdict(float)
        self.bigram_counts = defaultdict(float)
        self.log_probs = defaultdict(lambda : defaultdict(lambda: None))
        self.smoothing = smoothing
        self.unigram_weight = 0.5
        self.bigram_weight = 0.5
        self.training_set_length = 0

        self.EstimateBigrams(training_corpus)


    def EstimateBigrams(self, training_corpus):
      self.__set_counts(training_corpus)
      self.__set_probs()


    def CheckDistribution(self):
      for unigram in self.unigram_counts:
        if unigram == end_token: continue
        prob_sum = sum( exp(x) for x in self.log_probs[unigram].values() )
        assert abs(prob_sum - 1.0) < 0.001, 'invalid probabilities in log_probs'


    def Perplexity(self, test_corpus):
      perplexity = float()
      divisor = 0

      for line in test_corpus:
        for i, token in enumerate(line):
          if token == end_token: continue

          probability = self.__bigram_prob(token, line[i+1])

          if probability is not None:
            perplexity += probability
            divisor += 1
          else:
            return float('inf')

      return exp(-(perplexity/divisor))


    def __bigram_prob(self, token1, token2):

      if self.smoothing == 'laplace':
        log_prob = self.log_probs[token1][token2]
        return log_prob if type(log_prob) is float else \
          log(1 / (self.unigram_counts[token1] + self.vocabulary_size))

      elif self.smoothing == 'simple linear':
        return self.__simp_lin_interp(token1, token2)
      elif self.smoothing == 'deleted interpolation':
        return self.__simp_lin_interp(token1, token2)
      else:
        return self.log_probs[token1][token2]


    def __simp_lin_interp(self, token1, token2):
      unigram_prob = self.unigram_counts[token2] / self.training_set_length
      log_prob = self.log_probs[token1][token2]
      bigram_prob = exp(
        log_prob if type(log_prob) is float else float('-inf')
      )

      return log((self.bigram_weight * bigram_prob) + (self.unigram_weight * unigram_prob))


    def set_deleted_interp_weight(self, held_out_set):
      bigram_weight, unigram_weight = float(), float()
      corpus_unigram_count, \
      corpus_bigram_count, \
      set_length = self.__count(held_out_set)

      for (token1, token2) in self.bigram_counts:
        bigram_case_divisor = (corpus_unigram_count[token1] - 1)

        bigram_case = \
          (corpus_bigram_count[(token1, token2)] - 1) \
          / bigram_case_divisor if bigram_case_divisor > 0 else 0

        unigram_case = \
          (corpus_unigram_count[token2] - 1) \
          / (set_length - 1)

        if bigram_case > unigram_case:
          bigram_weight += corpus_bigram_count[(token1, token2)]
        else:
          unigram_weight += corpus_bigram_count[(token1, token2)]

      weight_sum = bigram_weight + unigram_weight

      self.unigram_weight, self.bigram_weight = \
        (unigram_weight/weight_sum , bigram_weight/weight_sum)

      return (self.unigram_weight, self.bigram_weight)


    def __set_counts(self, corpus):
      self.unigram_counts, \
      self.bigram_counts, \
      self.training_set_length = self.__count(corpus)

    def __count(self, corpus):
      corpus_length = 0
      unigram_count, bigram_count = defaultdict(float), defaultdict(float)

      for line in corpus:
        line_length = len(line)
        for i, word in enumerate(line):
          unigram_count[word] += 1
          corpus_length += 1
          if i < line_length-1: bigram_count[ (word, line[i+1]) ] += 1

      return (unigram_count, bigram_count, corpus_length)


    def __set_probs(self):
      for (word1, word2) in self.bigram_counts:

        if self.smoothing == 'laplace':
          probability = \
            (self.bigram_counts[(word1,word2)] + 1) \
            / (self.unigram_counts[word1] + self.vocabulary_size)
        else:
          probability = \
            self.bigram_counts[(word1,word2)] \
            / self.unigram_counts[word1]

        self.log_probs[word1][word2] = log(probability)


def main():
    training_set = brown.sents()[:50000]
    held_out_set = brown.sents()[-6000:-3000]
    test_set = brown.sents()[-3000:]
    
    vocabulary = makeVocab(training_set)

    """ Transform the data sets by eliminating unknown words and adding sentence boundary 
    tokens.
    """
    training_set_prep = PreprocessText(training_set, vocabulary)
    held_out_set_prep = PreprocessText(held_out_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)


    # Print the first sentence of each data set.
    print training_set_prep[0]
    print held_out_set_prep[0]
    print test_set_prep[0]

    # Estimate a bigram_lm object, check its distribution, compute its perplexity.
    bigram_lm = BigramLM(vocabulary, training_set_prep)
    bigram_lm.CheckDistribution()

    print "Perplexity of Test Set without smoothing:"
    print bigram_lm.Perplexity(test_set_prep)

    print "Perplexity of Test Set with Laplace smoothing:"
    print BigramLM(vocabulary, training_set_prep, 'laplace').Perplexity(test_set_prep)

    print "Perplexity of Test Set with Simple Linear Interpolation weights of 1/2:"
    print BigramLM(vocabulary, training_set_prep, 'simple linear').Perplexity(test_set_prep)

    """ Estimate interpolation weights using the deleted interpolation algorithm on the 
    held out set and print out.
    """ 
    bigram_lm = BigramLM(vocabulary, training_set_prep, 'deleted interpolation')
    bigram_lm.set_deleted_interp_weight(held_out_set)

    print "Unigram Weight:"
    print bigram_lm.unigram_weight
    print "Bigram Weight:"
    print bigram_lm.bigram_weight

    print "Perplexity of Test Set with Simple Linear Interpolation weights calculated by Deleted Interpolation:"
    print bigram_lm.Perplexity(test_set_prep)

if __name__ == "__main__": 
    main()