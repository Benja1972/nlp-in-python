import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import string
from os.path import join 
import os
import pickle

import pandas as pd

DATA = 'transcripts'



data = pd.read_pickle(join(DATA,"dtm_stop.pkl"))

from gensim import matutils, models
import scipy.sparse
tdm = data.transpose()


# We're going to put the term-document matrix into a new gensim format, 
# from df --> sparse matrix --> gensim corpus
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)



# Gensim also requires dictionary of the all terms and their respective 
# location in the term-document matrix
cv = pickle.load(open(join(DATA,"cv_stop.pkl"), "rb"))
id2word = dict((v, k) for k, v in cv.vocabulary_.items())

# Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),
# we need to specify two other parameters as well - the number of topics and the number of passes
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=4, passes=10)
lda.print_topics()

from nltk import word_tokenize, pos_tag

def nouns(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'S'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized, lang='rus') if is_noun(pos)] 
    return ' '.join(all_nouns)

def nouns(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'S' or  pos[:2] == 'A'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized, lang='rus') if is_noun(pos)] 
    return ' '.join(all_nouns)



data_clean = pd.read_pickle(join(DATA,"clean.pkl"))

data_nouns = pd.DataFrame(data_clean.text.apply(nouns))


## Document-Term Matrix
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")

from sklearn.feature_extraction.text import CountVectorizer


cvn = CountVectorizer(stop_words=russian_stopwords)
data_cvn = cvn.fit_transform(data_clean.text)

data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())

data_dtmn.index = data_clean.index



# Create the gensim corpus
corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))

# Create the vocabulary dictionary
id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())



# Let's try 4 topics
ldan = models.LdaModel(corpus=corpusn, num_topics=4, id2word=id2wordn, passes=10)
ldan.print_topics()
