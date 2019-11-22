import string
from os.path import join 
import os
import pickle

import pandas as pd

DATA = 'transcripts'

from nltk import word_tokenize, pos_tag


def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'S' or  pos[:2] == 'A'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized, lang='rus') if is_noun(pos)] 
    return ' '.join(all_nouns)


# Clean text to keep only nouns and adjectives
data_clean = pd.read_pickle(join(DATA,"clean.pkl"))
data_nouns = pd.DataFrame(data_clean.text.apply(nouns_adj))


## Rebuild Document-Term Matrix
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

russian_stopwords = stopwords.words("russian")


cv = CountVectorizer(stop_words=russian_stopwords)
data_cv = cv.fit_transform(data_clean.text)

data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index

from gensim import matutils, models
import scipy.sparse

# We're going to put the term-document matrix into a new gensim format, 
# from df --> sparse matrix --> gensim corpus
corpus = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtm.transpose()))

# Create the vocabulary dictionary
id2word = dict((v, k) for k, v in cv.vocabulary_.items())



# Let's try 4 topics
lda = models.LdaModel(corpus=corpus, num_topics=4, id2word=id2word, passes=80)
print('Topics = ', lda.print_topics())


#Let's take a look at which topics each transcript contains
corpus_transformed = lda[corpus]

top_list = list(zip([a for [(a,b)] in corpus_transformed], data_dtm.index))

print('Topics by authors = ', top_list)
