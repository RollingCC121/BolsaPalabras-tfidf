"""Bolsa de palabras TFIDF

#Preparacion del texto
1. Limpieza del Texto
2. Eliminar Stopwords
3. Reducir las palabras a  las raíces o lemas
4. Visualización del texto
5. Representación TF*IDF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
nltk.download('popular')

data = pd.read_csv("trenesValencia.csv", sep = ",", encoding = "latin-1")
data.head()

data.info()

#convertir minusculas, limpiar y tokeizar
from nltk.tokenize import word_tokenize

def tokenizar(texto):
  tokens = word_tokenize(texto)
  words = [w.lower() for w in tokens if w.isalnum()]
  return words

data['tokens'] = data['Llamada'].apply(lambda x: tokenizar(x))
data.head()

"""#2.Eliminar stopwords y otras palabras"""

#Eliminamos stopwords y otras palabras
from nltk.corpus import stopwords

sw= stopwords.words('spanish')
sw.append("https") #adiciona nuevas palabras a la lista de stopwords

def limpiar_stopwords(lista):
  clean_tokens = lista[:]
  for token in lista:
    if token in sw:
      clean_tokens.remove(token)
  return clean_tokens

# Limpiamos los tokens
data['sin_stopwords'] = data['tokens'].apply(lambda x: limpiar_stopwords(x))
data.head()

"""#3. Reducción a la raiz"""

#Reducción a la raíz (Stemming)
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')

def stem_tokens(lista):
  lista_stem = []
  for token in lista:
    lista_stem.append(stemmer.stem(token))
  return lista_stem

data['stemming'] = data['sin_stopwords'].apply(lambda x: stem_tokens(x))
data.head()

"""#4. Visualización del texto"""

#Nube de palabras
from wordcloud import WordCloud

#lista_palabras = data["stemming"].tolist()
lista_palabras = data["sin_stopwords"].tolist()
tokens = [keyword.strip() for sublista in lista_palabras for keyword in sublista]
texto= ' '.join(tokens)
wc = WordCloud(background_color="white", max_words=1000, margin=0)
wc.generate(texto)
wc.to_file("nube1.png")
plt.figure(figsize=(15,15))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

#Gráfica de palabras mas frecuentes de stemming
freq = nltk.FreqDist(tokens)
plt.figure(figsize=(8, 8))
freq.plot(20, cumulative=False)

#Representación en vector de características tf*idf

from sklearn.feature_extraction.text import TfidfVectorizer

def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)

X = tfidf.fit_transform(data['stemming']) #stemming, lemmatization
data_tfidf=pd.DataFrame(X.todense(),columns=tfidf.get_feature_names())
data_tfidf
