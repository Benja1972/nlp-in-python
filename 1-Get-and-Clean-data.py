import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import string
from os.path import join 
import os
import pickle


urls = {'Bowt':'https://snob.ru/entry/185057/',
    'Mlshtn':'https://snob.ru/entry/185060/',
    'Kuval':'https://snob.ru/entry/185010/',
    'Dvdv':'https://snob.ru/entry/184951/',
    'Prav':'https://snob.ru/entry/184852/',
    'Mrz':'https://snob.ru/news/184870/',
    'Znam':'https://snob.ru/entry/184780/',
    'Mikol':'https://snob.ru/entry/184642/',
    'Inoz':'https://snob.ru/entry/184764/',
    'Mashk':'https://snob.ru/entry/184611/'
    }



full_names =    ['Георгий Бовт',
                'Илья Мильштейн',
                'Станислав Кувалдин',
                'Иван Давыдов',
                'Ксения Праведная',
                'Ольга Морозова',
                'Анна Знаменская',
                'Дарья Миколайчук',
                'Владислав Иноземцев',
                'Диана Машкова']

DATA = 'transcripts'

if not os.path.exists(DATA):
    os.makedirs(DATA)
    print("Directory " , DATA ,  " created ")
else:    
    print("Directory " , DATA ,  " already exists")    

# Scrape transcript data
def url_to_text(url):
    '''Returns transcript data from snob.ru.'''
    page = requests.get(url).text
    soup = BeautifulSoup(page, "lxml")
    text = [p.text for p in soup.find(class_="text entry__text js-mediator-article").find_all('p')]
    print(url+" --- DONE")
    return text




data = pd.DataFrame(index=urls.keys(), columns=['transcripts'])

for name, url in urls.items():
    data.loc[name]['transcripts'] = url_to_text(url)


data['text'] = [' '.join(rw['transcripts']) for ind,rw in data.iterrows()]  
data['full_name'] = full_names                                                                    
data.to_pickle(join(DATA,"transcripts.pkl"))



def clean_round(text):
    '''Clean first'''
    text = text.lower()
    text = text.replace('ё','е')
    text = text.replace('Ё','Е')
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("—", "", text)
    text = re.sub("/", " ", text)
    text = re.sub('[‘’“”…»«]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub("[a-z]+","", text)
    text = re.sub('\xa0', ' ', text)
    text = re.sub(' +', ' ', text) 
    return text   

cln = lambda x: clean_round(x)

data_clean = pd.DataFrame(data.text.apply(cln))

data_clean['full_name'] = full_names
data_clean.to_pickle(join(DATA,"clean.pkl"))


## Document-Term Matrix
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stopwords_russian = set(stopwords.words("russian"))

cv = CountVectorizer(stop_words=stopwords_russian)
data_cv = cv.fit_transform(data_clean.text)

data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())

data_dtm.index = data_clean.index

# Save data 
data_dtm.to_pickle(join(DATA,"dtm.pkl"))
pickle.dump(cv, open(join(DATA,"cv.pkl"), "wb"))
