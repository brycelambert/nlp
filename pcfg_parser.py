import sys, re
import nltk
from nltk.corpus import treebank
from collections import defaultdict
from nltk import induce_pcfg
from nltk.grammar import Nonterminal
from nltk.tree import Tree
from math import exp, pow, log
from IPython import embed

unknown_token = "<UNK>"  # unknown word token.

def makeVocab(corpus):
  corpus_hash = defaultdict(int)

  for line in corpus:
    for word in line.leaves(): corpus_hash[word] += 1

    vocab = set(
      (k) for k,v in corpus_hash.items() if v > 1 )

    vocab.update(unknown_token)
    
  return vocab

def PreprocessText(corpus, vocab):
  output = list(corpus)

  for tree in corpus:
    for node in PreterminalNodes(tree):
      if node[0] not in vocab: node[0] = unknown_token

  return output

def makePCFG(training_set):
  productions = []
  for line in training_set: productions += line.productions()
  return induce_pcfg( Nonterminal('S'), productions )


""" Removes all function tags e.g., turns NP-SBJ into NP.
"""         
def RemoveFunctionTags(tree):
    for subtree in tree.subtrees():  # for all nodes of the tree
        # if it's a preterminal node with the label "-NONE-", then skip for now
        if subtree.height() == 2 and subtree.label() == "-NONE-": continue
        nt = subtree.label()  # get the nonterminal that labels the node
        labels = re.split("[-=]", nt)  # try to split the label at "-" or "="
        if len(labels) > 1:  # if the label was split in two e.g., ["NP", "SBJ"]
            subtree.set_label(labels[0])  # only keep the first bit, e.g. "NP"

""" Return true if node is a trace node.
"""         
def IsTraceNode(node):
    # return true if the node is a preterminal node and has the label "-NONE-"
    return node.height() == 2 and len(node) == 1 and node.label() == "-NONE-"

""" Deletes any trace node children and returns true if all children were deleted.
"""
def RemoveTraces(node):
    if node.height() == 2:  # if the node is a preterminal node
        return False  # already a preterminal, cannot have a trace node child.
    i = 0
    while i < len(node):  # iterate over the children, node[i]
        # if the child is a trace node or it is a node whose children were deleted
        if IsTraceNode(node[i]) or RemoveTraces(node[i]): 
            del node[i]  # then delete the child
        else: i += 1
    return len(node) == 0  # return true if all children were deleted
    
""" Preprocessing of the Penn treebank.
"""
def TreebankNoTraces():
    tb = []
    for t in treebank.parsed_sents():
        if t.label() != "S": continue
        RemoveFunctionTags(t)
        RemoveTraces(t)
        t.collapse_unary(collapsePOS = True, collapseRoot = True)
        t.chomsky_normal_form()
        tb.append(t)
    return tb
        
""" Enumerate all preterminal nodes of the tree.
""" 
def PreterminalNodes(tree):
    for subtree in tree.subtrees():
        if subtree.height() == 2:
            yield subtree
    
""" Print the tree in one line no matter how big it is
    e.g., (VP (VB Book) (NP (DT that) (NN flight)))
"""         
def PrintTree(tree):
    if tree.height() == 2: return "(%s %s)" %(tree.label(), tree[0])
    return "(%s %s)" %(tree.label(), " ".join([PrintTree(x) for x in tree]))
    
class InvertedGrammar:
  def __init__(self, pcfg):
    self._pcfg = pcfg
    self._r2l = defaultdict(list)  # maps RHSs to list of LHSs
    self._r2l_lex = defaultdict(list)  # maps lexical items to list of LHSs
    self.BuildIndex()  # populates self._r2l and self._r2l_lex according to pcfg
			
  def PrintIndex(self, filename):
    f = open(filename, "w")
    for rhs, prods in self._r2l.iteritems():
    	f.write("%s\n" %str(rhs))
    	for prod in prods:
    		f.write("\t%s\n" %str(prod))
    	f.write("---\n")
    for rhs, prods in self._r2l_lex.iteritems():
    	f.write("%s\n" %str(rhs))
    	for prod in prods:
    		f.write("\t%s\n" %str(prod))
    	f.write("---\n")
    f.close()
        
  def BuildIndex(self):
    for production in self._pcfg.productions():
      if production.is_lexical():
        self._r2l_lex[production.rhs()[0]].append(production)
      else:
        self._r2l[production.rhs()].append(production)

    self.PrintIndex('index')

            
  def Parse(self, sent):
    table = defaultdict(lambda: defaultdict(lambda: defaultdict((lambda: float('-inf')))))
    backpointers = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for j in range(0, len(sent)):
      token = sent[j]

      for a in self._r2l_lex[token]:
        table[j][j][a.lhs()] = a.logprob()

      for i in range(j-1, -1, -1):
        for k in range(i, j):
          for b, b_prob in table[i][k].iteritems():
            for c, c_prob in table[k+1][j].iteritems():
              for a in self._r2l[(b, c)]:
                #and greater than zeros?
                a_lhs = a.lhs()
                new_prob = a.logprob() + b_prob + c_prob
                if table[i][j][a_lhs] < new_prob:
                  table[i][j][a_lhs] = new_prob
                  backpointers[i][j][a_lhs] = (k, b, c)

    return (table, backpointers)

        
  @staticmethod
  def BuildTree(cky_table, sent, j, a=Nonterminal("S"), i=0):
    if i == j:
      return Tree(a.symbol(), [sent[i]])
    else:
      (k, b, c) = cky_table[i][j][a]
      return Tree(
        a.symbol(),
        [InvertedGrammar.BuildTree(cky_table, sent, k, b, i),
         InvertedGrammar.BuildTree(cky_table, sent, j, c, k+1)])

def build_buckets(data):
  bucket1, bucket2, bucket3, bucket4, bucket5 = [], [], [], [], []
  
  for sentence in data:
    sentence_len = len(sentence.leaves())
    if  0 < sentence_len < 10:
      bucket1.append(sentence)
    elif 10 <= sentence_len < 20:
      bucket2.append(sentence)
    elif 20 <= sentence_len < 30:
      bucket3.append(sentence)
    elif 30 <= sentence_len < 40:
      bucket4.append(sentence)
    elif sentence_len >= 40:
      bucket5.append(sentence)
    else:
      None

  return [bucket1, bucket2, bucket3, bucket4, bucket5]


def print_bucket(bucket, label, ig):
  print "Printing bucket", label
  predicted_out = open("test_%s"%label, "w")
  gold_out = open("gold_%s"%label, "w")

  for tree in bucket:
    sentence = tree.leaves()
    table, backpointers = ig.Parse(sentence)

    if Nonterminal("S") not in table[0][len(sentence)-1].keys():
      predicted_tree = False
    else:
      predicted_tree = ig.BuildTree(backpointers, sentence, len(sentence)-1)

    if predicted_tree:
      predicted_tree.un_chomsky_normal_form()
      if type(tree) == type(None): embed()
      predicted_out.write( PrintTree(predicted_tree) + '\n' )
      print "line down"
    else:
      predicted_out.write("\n")

    tree_out = tree
    tree_out.un_chomsky_normal_form()
    gold_out.write( PrintTree(tree_out) +'\n' )

  predicted_out.close()
  gold_out.close()

def main():
  treebank_parsed_sents = TreebankNoTraces()
  training_set = treebank_parsed_sents[:3000]
  test_set = treebank_parsed_sents[3000:]

  vocabulary = makeVocab(training_set)
  training_set_prep = PreprocessText(training_set, vocabulary)
  test_set_prep = PreprocessText(test_set, vocabulary)

  #Part 1
  print PrintTree( training_set_prep[0] )
  print PrintTree( test_set_prep[0] )

  #Part 2
  grammar = makePCFG(training_set_prep)
  np_nonterm_productions = grammar.productions(lhs=Nonterminal('NP'))
  np_nonterm_productions.sort(key = lambda x: x.prob(), reverse=True)

  print'Productions found for NP Terminal: ', len(np_nonterm_productions)
  print '...the 10 most likely being:'
  for production in np_nonterm_productions[:10]: print production

  # Part 3.1 (prints 'index')
  inverted_g = InvertedGrammar(grammar)

  #Part 3.2
  test_sent = "Terms were n't disclosed .".split()
  table, backpointers = inverted_g.Parse(test_sent)
  print 'Log probability of test sentence:', table[0][4][Nonterminal('S')]

  #Part 3.3
  print "Tree for test sentence:"
  print inverted_g.BuildTree(backpointers, test_sent, len(test_sent)-1)

  #Part 3.4
  print "Bucket sizes:"
  buckets = build_buckets(test_set_prep)

  for i in range (0, 5):
    print "Bucket", i+1,":", len(buckets[i])

  # Part 3.5 Tree Printer (print_bucket calls Parse and writes files)
  buckets_to_print = 5
  for i in range(0, buckets_to_print): print_bucket(buckets[i], i+1, inverted_g)

  #Part 3.5 Evalb


    
if __name__ == "__main__": 
  main()
