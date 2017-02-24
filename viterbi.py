from IPython import embed

import sys
from collections import defaultdict
from math import log, exp

from nltk.corpus import treebank
from nltk.tag.util import untag  # Untags a tagged sentence. 

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.

""" Remove trace tokens and tags from the treebank as these are not necessary.
"""
def TreebankNoTraces():
    return [[x for x in sent if x[1] != "-NONE-"] for sent in treebank.tagged_sents()]

######################################################
def makeVocab(corpus):
  corpus_hash = defaultdict(int)

  for line in corpus:
    for word in line: corpus_hash[ word[0] ] += 1

  vocab = set(
    (k) for k,v in corpus_hash.items() if v > 1 )

  vocab.update({ unknown_token, start_token, end_token })
  return vocab

def PreprocessText(corpus, vocab):
  output = list(corpus)

  for line in output:
    line.insert(0, (start_token, start_token) )
    line.append( (end_token, end_token) )
    
    for i, tagged_word in enumerate(line):
      if tagged_word[0] not in vocab: line[i] = (unknown_token, line[i][1])

  return output

########################################################


        
class BigramHMM:
    def __init__(self):
        self.transitions = defaultdict(lambda: -float('inf'))
        # the A matrix of the HMM: a_{ij} = P(t_j | t_i)

        self.emissions = defaultdict(float)
        # the B matrix of the HMM: b_{ii} = P(w_i | t_i)

        self.dictionary = defaultdict(set)
        # a dictionary that maps a word to the set of possible tags
        
    def Train(self, training_set):
        """ 
        1. Estimate the A matrix a_{ij} = P(t_j | t_i)
        2. Estimate the B matrix b_{ii} = P(w_i | t_i)
        3. Compute the tag dictionary
        """
        tag_counter = defaultdict(int)

        for line in training_set:
          for token_i, tagged_token in enumerate(line):
            token = tagged_token[0]
            tag = tagged_token[1]
            self.dictionary[token].add(tag)
            tag_counter[tag] += 1
            self.emissions[tagged_token] += 1

            if tagged_token[0] is not end_token:
              next_tag = line[token_i + 1][1]
              if (tag, next_tag) not in self.transitions: self.transitions[tag, next_tag] = 0.0
              self.transitions[tag, next_tag] += 1

        for tagged_unigram in self.emissions:
          self.emissions[tagged_unigram] = log( self.emissions[tagged_unigram] / tag_counter[tagged_unigram[1]] )

        for tag_bigram in self.transitions:
          self.transitions[tag_bigram] = log( self.transitions[tag_bigram] / tag_counter[tag_bigram[0]] )

        return True

    def ComputePercentAmbiguous(self, data_set):
        ambiguous_token_count = 0.0
        token_count = 0.0

        for line in data_set:
          for tagged_token in line:
            if len(self.dictionary[tagged_token[0]]) > 1: ambiguous_token_count += 1
            token_count += 1

        print "UNK token tag count: ", len( self.dictionary[unknown_token] )
        print "UNK token tags: " + ', '.join(self.dictionary[unknown_token])

        return ambiguous_token_count / float(token_count) * 100
        
    def JointProbability(self, sent):
      output = 0.0

      for i in range(1, len(sent)):
        tagged_token = sent[i]
        prev_tagged_token = sent[i-1]
        output += self.emissions[tagged_token] + self.transitions[prev_tagged_token[1], tagged_token[1]]

      return exp(output)
        
    def Viterbi(self, sent):
      v_matrix = defaultdict(dict)
      backpointers = {}

      init_token = sent[1][0]
      tagged_init_state = self.dictionary[init_token]
      for state in tagged_init_state:
        v_matrix[1][state] = \
          self.transitions[start_token, state] + self.emissions[init_token, state]
        backpointers[state, 1] = start_token

      for t in range(2, len(sent)):
        token = sent[t][0]
        for state in self.dictionary[token]:
          max_v_matrix_key = max(v_matrix[t-1], key=lambda k: v_matrix[t-1][k] + self.transitions[k, state])
          v_matrix[t][state] = v_matrix[t-1][max_v_matrix_key] + self.emissions[token, state] + self.transitions[max_v_matrix_key,state]
          backpointers[state, t] = max_v_matrix_key

      tag = end_token
      output = [(end_token, end_token)]
      for i in range(1, len(sent)):
        offset = len(sent)-i
        tag = backpointers[tag, offset]
        output.insert(0, (sent[offset-1][0], tag))

      return output

        
    def Test(self, test_set):
      output = []
      for i, line in enumerate(test_set):
        output.append(self.Viterbi(line))
      return output



def MostCommonClassBaseline(training_set, test_set):
    training_pos_counts = defaultdict(lambda: defaultdict(int))
    output = []

    for line in training_set:
      for tagged_token in line:
        training_pos_counts[tagged_token[0]][tagged_token[1]] += 1

    for line in test_set:
      new_line = []
      for tagged_token in line:
        token = tagged_token[0]
        tag = max(training_pos_counts[token], key=lambda pos: training_pos_counts[token][pos])
        new_line.append( (token, tag) )

      output.append(new_line)

    return output

    
def ComputeAccuracy(test_set, test_set_predicted):
  sentence_accuracy = [0.0, 0.0]
  tagging_accuracy = [0.0,0.0]

  for index, line in enumerate(test_set):
    if line == test_set_predicted[index]: sentence_accuracy[0] += 1
    sentence_accuracy[1] += 1
  
    for token_i, token in enumerate(line):
      if token in [start_token, end_token]: continue

      if token == test_set_predicted[index][token_i]: tagging_accuracy[0] += 1
      tagging_accuracy[1] += 1

  print "Sentence accuracy: ", sentence_accuracy[0] / sentence_accuracy[1]
  print "Tagging accuracy: ", tagging_accuracy[0] / tagging_accuracy[1]

    
def main():
    treebank_tagged_sents = TreebankNoTraces()  # Remove trace tokens. 
    training_set = treebank_tagged_sents[:3000]  # This is the train-test split that we will use. 
    test_set = treebank_tagged_sents[3000:]
    vocabulary = makeVocab(training_set)

    training_set_prep = PreprocessText(training_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)

    # Print the first sentence of each data set.
    print " ".join(untag(training_set_prep[0]))  # See nltk.tag.util module.
    print " ".join(untag(test_set_prep[0]))

    # Estimate Bigram HMM from the training set, report level of ambiguity.
    bigram_hmm = BigramHMM()
    bigram_hmm.Train(training_set_prep)
    print "Percent tag ambiguity in training set is %.2f%%." %bigram_hmm.ComputePercentAmbiguous(training_set_prep)
    print "Joint probability of the first sentence is %s." %bigram_hmm.JointProbability(training_set_prep[0])
    
    # Implement the most common class baseline. Report accuracy of the predicted tags.
    test_set_predicted_baseline = MostCommonClassBaseline(training_set_prep, test_set_prep)
    print "--- Most common class baseline accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_baseline)

    # Use the Bigram HMM to predict tags for the test set. Report accuracy of the predicted tags.
    test_set_predicted_bigram_hmm = bigram_hmm.Test(test_set_prep)
    print "--- Bigram HMM accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_bigram_hmm)

if __name__ == "__main__": 
    main()
    