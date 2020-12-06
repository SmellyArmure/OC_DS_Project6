'''Takes the H matrix (topics/words) as a dataframe, extracts the n top words
and plots a wordcloud of the (n_top_words) top words for each topic.
'''

from wordcloud import WordCloud

def plot_wordclouds_from_H(H, n_top_words, n_rows=1, figsize=(18,8),
                             random_state=None):

    fig = plt.figure(figsize=figsize)
    wc = WordCloud(stopwords=None, background_color="black",
                   colormap="Dark2", max_font_size=150,
                   random_state=random_state)
    # boucle sur les thèmes
    for i, topic_name in enumerate(H.index,1):
        ser_10w_topic = H.loc[topic_name]\
            .sort_values(ascending=False)[0:n_top_words]
        wc.generate(' '.join(list(ser_10w_topic.index)))
        n_tot = H.index.shape[0]
        n_cols = (n_tot//n_rows)+((n_tot%n_rows)>0)*1
        ax = fig.add_subplot(n_rows,n_cols,i)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.title(topic_name, fontweight='bold')
        
    plt.show()


'''Takes a groupby made on a series of texts (non tokenized),
(-> for example : gb = df_desc_cat.groupby('category')['desc_clean'])
extracts the n top words and plots a wordcloud of the (n_top_words)
top words for each topic.
'''

import matplotlib.pyplot as plt
from wordcloud import WordCloud

def plot_wordclouds_from_gb(gb, n_top_words, n_rows=1, figsize=(18,8),
                             backgnd_color='black', cmap='Dark2',
                            random_state=None):

    fig = plt.figure(figsize=figsize)

    for i, tup in enumerate(gb,1):
        n_topic, ser_texts = tup
        # creation of a corpus of all the cleaned descriptions and product_names
        corpus = ' '.join(ser_texts.values)
        # tokenizing the words in the cleaned corpus
        tokenizer = nltk.RegexpTokenizer(r'[a-z]+')
        li_words = tokenizer.tokenize(corpus.lower())
        # counting frequency of each word
        ser_freq = pd.Series(nltk.FreqDist(li_words))

        wc = WordCloud(stopwords=None, background_color=backgnd_color,
                        colormap=cmap, max_font_size=150,
                        random_state=14)
        ser_topic = ser_freq\
            .sort_values(ascending=False)[0:n_top_words]
        wc.generate(' '.join(list(ser_topic.index)))

        n_tot = len(gb)
        n_cols = (n_tot//n_rows)+((n_tot%n_rows)>0)*1
        ax = fig.add_subplot(n_rows,n_cols,i)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.title(n_topic, fontweight='bold')


''' Takes a pd.Series containing the texts of each description
applies a preprocessing function if given (stopwords, stemming...)
then turn the descriptions in vectors (bow of tf-idf, depending on the avlue of
 tfidf_on)
 returns document term matrix as a dataframe and the list of new excluded words.
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_doc_terms_df(ser_desc, 
                         preproc_func=None,
                         preproc_func_params=None,
                         vec_params = {'min_df': 1},
                         tfidf_on=False,
                         print_opt=False):

    # ---- Apply a preprocessing function prior to vectorization
    if preproc_func is not None:
        ser_desc = ser_desc.apply(lambda x: preproc_func(x,
                                                         **preproc_func_params))
        ser_desc = ser_desc.apply(lambda x: ' '.join(x))
    else:
        ser_desc = ser_desc
    
    # ---- Vectorization of each of the texts (row)
    if tfidf_on:
        # TF-IDF matrix
        vec = TfidfVectorizer(**vec_params)
    else:
        # BOW matrix (count)
        vec = CountVectorizer(**vec_params)

    doc_term = vec.fit_transform(ser_desc)
    if print_opt:
        print( "Created %d X %d doc_term matrix" % (doc_term.shape[0],
                                                    doc_term.shape[1]))

    # ---- Vocabulary of the document_term matrix
    doc_term_voc = vec.get_feature_names()
    if print_opt:
        print("Vocabulary has %d distinct terms" % len(doc_term_voc))

    # ---- Get the list of the new stop-words
    new_sw = vec.stop_words_
    if print_opt:
        print("Old stop-words list has %d entries" % len(sw) )
        print("New stop-words list has %d entries" % len(new_sw))

    doc_term_df = pd.DataFrame(doc_term.todense(),
                index=ser_desc.index, # each item
                columns=doc_term_voc) # each word

    # document term matrix as a dataframe and the list of new excluded words
    return doc_term_df, new_sw


'''
Takes a vectorized matrix (dataframe) of the documents
(Document-trem matrix: BOW or tf-idf... documents(rows) x words (columns))
and returns the projected vectors in the form of a dataframe
(words (rows) x w2v dimensions(columns))
'''

def proj_term_doc_on_w2v(term_doc_df, w2v_model, print_opt=False):

    # Checking the number of words of our corpus existing in the wiki2vec dictionary
    li_common_words = []
    for word in term_doc_df.columns:
        word_ = w2v_model.get_word(word)
        if word_ is not None:
            li_common_words.append(word)
    if print_opt:
        print(f"The w2v dictionary contains {len(li_common_words)} words out of \
the {term_doc_df.shape[1]} existing in our descriptions,\ni.e. \
{round(100*len(li_common_words)/term_doc_df.shape[1],1)}% of the whole vocabulary.")

    # extracting each of the word vectors
    word_vectors_df = pd.DataFrame()
    for word in li_common_words:
        word_vectors_df[word] = w2v_model.get_word_vector(word)
    word_vectors_df = word_vectors_df.T
    word_vectors_df.columns = ['dim_'+str(i)\
                               for i in range(word_vectors_df.shape[1])]

    # projection of the Document_terms matrix on the wiki2vec
    w2v_emb_df = term_doc_df[li_common_words].dot(word_vectors_df)

    return w2v_emb_df



''' from a sentence, containing words (document):
- tokenizes the words if only composed of alphanumerical data,
- removes stopwords if list is given (stopwords)
- stems the words if stemmer given
NB: This pre-processing function can be used to prepare data for Word2Vec
'''
from nltk.stem.snowball import EnglishStemmer
import spacy
import nltk
nltk.download('averaged_perceptron_tagger')

def tokenize_clean(document, stopwords=None, keep_tags=None, # ('NN' or 'JJ')
                   lemmatizer=None, stemmer=None):
    # 1 - tokenizing the words in each description
    tokenizer = nltk.RegexpTokenizer(r'[A-Za-z]+')
    li_words = tokenizer.tokenize(document)
    # 2 - lower case
    li_words = [s.lower() for s in li_words]
    # 3 - keep only certain tags
    if keep_tags is not None:
        li_words = [word for word,tag in nltk.pos_tag(li_words)\
            if tag in keep_tags]
    if stopwords is None: stopwords=[]
    # 4 - lemmatizing or stemming
    if lemmatizer is not None:
        lem_doc = lemmatizer(' '.join(li_words))
        li_words = [token.lemma_ for token in lem_doc]
    elif stemmer is not None:
        li_words = [stemmer.stem(s) for s in li_words]
    # 5 - removing stopwords
    li_words = [s for s in li_words if s not in stopwords]

    return li_words


''' Builds a customizable NLP column_transformer which parameters
can be optimized in a GridSearchClust
'''
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
# from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import EnglishStemmer
import spacy
import nltk
nltk.download('averaged_perceptron_tagger')

import numpy as np
import pandas as pd
from wikipedia2vec import Wikipedia2Vec

from wikipedia2vec import Wikipedia2Vec

class CustNLPTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, keep_tags=None, stemmer=None,
                 lemmatizer=None, min_df=0, max_df=1.0, max_features=None,
                 tfidf_on=False, ngram_range=(1,1), binary=False,
                 w2v=None, path_wiki2vec="/content/enwiki_20180420_100d.pkl",
                 pname_weight=0.5):
        
        self.stopwords = stopwords
        self.keep_tags = keep_tags
        self.lemmatizer = lemmatizer
        self.stemmer = stemmer
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.tfidf_on = tfidf_on
        self.ngram_range = ngram_range
        self.binary = binary
        self.w2v = w2v
        self.path_wiki2vec = path_wiki2vec
        self.pname_weight = pname_weight
        self.preproc_func_params={'stopwords': self.stopwords,
                                  'keep_tags': self.keep_tags,
                                  'lemmatizer': self.lemmatizer,
                                  'stemmer': self.stemmer}
        self.vec_params = {'min_df': self.min_df,
                           'max_df': self.max_df,
                           'max_features': self.max_features,
                           'ngram_range': self.ngram_range,
                           'binary': self.binary}


        # all preprocessing params to None (faster)
        if set([v for k, v in self.preproc_func_params.items()]) == set([None]):
            self.preproc_func = None
        else: # else tokenize_clean private function will be called
            self.preproc_func = self.__tokenize_clean

    # "private" method to prepropcess data inside '__compute_doc_terms_df'

    def __tokenize_clean(self, document, stopwords=None, keep_tags=None, # ('NN','JJ')
                    lemmatizer=None, stemmer=None):
        
        # 1 - tokenizing the words in each description
        tokenizer = nltk.RegexpTokenizer(r'[A-Za-z]+')
        li_words = tokenizer.tokenize(document)
        # 2 - lower case
        li_words = [s.lower() for s in li_words]
        # 3 - keep only certain tags
        if keep_tags is not None:
            li_words = [word for word,tag in nltk.pos_tag(li_words)\
                if tag in keep_tags]
        if stopwords is None: stopwords=[]
        # 4 - lemmatizing or stemming
        if lemmatizer is not None:
            lem_doc = lemmatizer(' '.join(li_words))
            li_words = [token.lemma_ for token in lem_doc]
        elif stemmer is not None:
            li_words = [stemmer.stem(s) for s in li_words]
        # 5 - removing stopwords
        li_words = [s for s in li_words if s not in stopwords]

        return li_words


    # "private" method to be used to apply transformation and get a df
    def __compute_doc_terms_df(self, ser_desc, preproc_func=None,
                               preproc_func_params=None, vec_params=None,
                               tfidf_on=None, vec=None):
        # ---- Apply a stemming or lemmatization prior to vectorization
        if preproc_func is not None:
            ser_desc = ser_desc.apply(lambda x: \
                                      preproc_func(x, **preproc_func_params))
            ser_desc = ser_desc.apply(lambda x: ' '.join(x))
        else:
            ser_desc = ser_desc
        # ---- Vectorization of each of the texts (row)
        if vec is None: # if no trained vectorized given
            if tfidf_on:
                # TF-IDF matrix
                vec = TfidfVectorizer(**vec_params)
            else:
                # BOW matrix (count)
                vec = CountVectorizer(**vec_params)
            vec.fit(ser_desc)
        else: # if a vectorizer is given
            try: # test if it is already fitted
                check_is_fitted(vec, attributes=None, all_or_any='any')
            except NotFittedError as e:
                vec.fit(ser_desc)
                print("Warning: 'vec' was not fitted -> has been fitted with 'df_desc'")
        doc_term = vec.transform(ser_desc)
        # ---- Vocabulary of the document_term matrix
        doc_term_voc = vec.get_feature_names()
        # # ---- Get the list of the new stop-words
        # new_sw = vec.stop_words_
        doc_term_df = pd.DataFrame(doc_term.todense(),
                                   index=ser_desc.index, # each item
                                   columns=doc_term_voc) # each word
        # document term matrix as a dataframe and the fitted vectorizer
        return doc_term_df, vec

    def fit(self, X, y=None):
        # nothing to fit - only set dictionaries (if a set_params had been run...)
        self.preproc_func_params={'stopwords': self.stopwords,
                                  'keep_tags': self.keep_tags,
                                  'lemmatizer': self.lemmatizer,
                                  'stemmer': self.stemmer}
        self.vec_params = {'min_df': self.min_df,
                           'max_df': self.max_df,
                           'max_features': self.max_features,
                           'ngram_range': self.ngram_range,
                           'binary': self.binary}
        return self

    def transform(self, X, y=None):  # returns a dataframe

        # X must be splitted in two parts : X_desc and X_pname
        X_desc, X_pname = X.iloc[:, 0], X.iloc[:, 1]

        # tranformation of X_pname into a custom BOW
        df_pname_trans, vec_fitted = self.__compute_doc_terms_df(\
                     ser_desc=X_pname,
                     preproc_func=self.preproc_func, 
                     preproc_func_params=self.preproc_func_params,
                     vec_params=self.vec_params,
                     tfidf_on=self.tfidf_on,
                     vec=None) # vec not fitted yet

        # tranformation of X_desc into a custom BOW (vec fitted with desc)
        df_desc_trans, _ = self.__compute_doc_terms_df(\
                     ser_desc=X_desc,
                     preproc_func=self.preproc_func,
                     preproc_func_params=self.preproc_func_params,
                     vec=vec_fitted) # vec fitted on the descriptions
        
        # Mix the X_desc and X_pname BOWs into one BOW (weight)

        df_trans = (df_desc_trans.mul(1-self.pname_weight,
                                      fill_value=0))\
                    .add(df_pname_trans.mul(self.pname_weight,
                                            fill_value=0),
                        fill_value=0)
        # if word_embedding is enabled, projection of the BOW on a given w2v
        if self.w2v:
            wiki2vec = Wikipedia2Vec.load(self.path_wiki2vec)
            df_trans = proj_term_doc_on_w2v(df_trans, wiki2vec,
                                            print_opt=False)
        return df_trans

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)




''' Builds a topics modeler which parameters (model, number of topics)
can be optimized in a GridSearchClust.
.transform: returns the DOCUMENTS/TOPICS matrix
.predict: returns the list of the most probable topic for each document
NB: takes a dataframe as X.
'''
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd

class TopicsModeler(BaseEstimator):

    def __init__(self, n_model='nmf', n_components=7, random_state=None):#, model_params):

        self.n_model = n_model
        self.n_components = n_components
        self.random_state = random_state
        # self.model_params = model_param

        # Model name -> object
        self.dict_models = {'lsa': TruncatedSVD(),
                            'nmf': NMF(init="nndsvd"),
                            'lda': LDA()}

        # Instantiate the model
        try:
            self.model = self.dict_models[self.n_model]#.set_params(*self.model_params)
        except:
            print(f"ERROR: {self.n_model} is an unknown topics modeliser. \n\
Please, choose between 'nmf', 'lda' and 'lsa'")

    def fit(self, X, y=None):

        # Re-Instantiate the model
        try:
            self.model = self.dict_models[self.n_model]#.set_params(*self.model_params)
        except:
            print(f"ERROR: {self.n_model} unknown topics modeliser. \n\
Please, choose between 'nmf', 'lda' and 'lsa'")

        # Set the parameters
        self.model.set_params(n_components = self.n_components,
                              random_state = self.random_state)

        # Fit the model
        self.model.fit(X)

        return self

    def __compute_DOC_TOP_matrix(self, X, y=None): # DOCUMENTS/TOPICS Matrix
    # actualization of n_components
        self.n_components = self.model.transform(X.values).shape[1]
        self.W = pd.DataFrame(self.model.transform(X.values),
                              index=X.index, # documents
                              columns=['topic_'+str(i)\
                                       for i in range(1,self.n_components+1)]) # topics

    def __compute_TOP_WORDS_matrix(self, X, y=None): # TOPICS/WORDS Matrix

        self.H = pd.DataFrame(self.model.components_, 
                              index=['topic_'+str(i)\
                                     for i in range(1,self.n_components+1)], # topics
                              columns=X.columns) # words

    def transform(self, X, y=None):  # to get the df of the DOC/TOPICS matrix

        self.__compute_DOC_TOP_matrix(X)
        self.__compute_TOP_WORDS_matrix(X)

        # Converting topics scores to best cluster label (higher val column)
        ser_res = self.W.idxmax(1)

        return self.W

    def predict(self, X, y=None):  # to get a ser of the best label

        self.__compute_DOC_TOP_matrix(X)
        self.__compute_TOP_WORDS_matrix(X)

        # Converting topics scores to best cluster label (higher val column)
        ser_res = self.W.idxmax(1)

        return ser_res

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X, y)


