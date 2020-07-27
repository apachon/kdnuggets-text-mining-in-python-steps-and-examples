############################################################################################################
#### Tokenization
############################################################################################################
# La tokenización es el primer paso en la NLP. 
# Es el proceso de romper el string en fichas que a su vez son pequeñas estructuras o unidades. 
# La Tokenización implica tres pasos:
#       romper una oración compleja en palabras
#       comprender la importancia de cada palabra con respecto a la oración 
#       finalmente producir una descripción estructural en una oración de entrada.

# Importing necessary library
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
import os
import nltk.corpus
# sample text for performing tokenization
text = "In Brazil they drive on the right-hand side of the road. Brazil has a large coastline on the eastern side of South America"
# importing word_tokenize from nltk
from nltk.tokenize import word_tokenize
# Passing the string text into word tokenize for breaking the sentences
token = word_tokenize(text)
token
print(token)

# Finding frequency distinct in the text
# finding the frequency distinct in the tokens
# Importing FreqDist library from nltk and passing token into FreqDist
from nltk.probability import FreqDist
fdist = FreqDist(token)
print(fdist)

# To find the frequency of top 10 words
fdist1 = fdist.most_common(10)
print(fdist1)

############################################################################################################
#### Stemming
############################################################################################################
# El "stemm" suele referirse a la normalización de las palabras en su forma base o forma de raíz.
# Aquí, tenemos esperó esperadas, esperando y espera. Aquí la palabra raíz es "espera". 
# Hay dos métodos en Stemming, a saber:
#       Porter Stemming (elimina las terminaciones morfológicas e inflexionales comunes de las palabras)
#       Lancaster Stemming (un algoritmo de stemming más agresivo).
# Importing Porterstemmer from nltk library

# Checking for the word ‘giving’ 
from nltk.stem import PorterStemmer
pst = PorterStemmer()
pst.stem("waiting")
print(pst.stem("waiting"))

# Checking for the list of words
stm = ["waited", "waiting", "waits"]
for word in stm :
   print(word+ ":" +pst.stem(word))

# Importing LancasterStemmer from nltk
from nltk.stem import LancasterStemmer
lst = LancasterStemmer()
stm = ["giving", "given", "given", "gave"]
for word in stm :
 print(word+ ":" +lst.stem(word))

############################################################################################################
#### Lemmatization
############################################################################################################
# Es el proceso de convertir una palabra a su forma base. La diferencia entre la stemming y la lemmatization es que
#  la lemmatización considera el contexto y convierte la palabra en su forma básica significativa, mientras que el stemming 
# sólo elimina los últimos caracteres, lo que a menudo conduce a significados incorrectos y errores de ortografía.

# Por ejemplo, la lemmatization identificaría correctamente la forma básica de "caring" a "care", mientras que la derivación 
# cortaría la parte "ing" y la convertiría en "coche".

# La lemmatization puede ser implementada en Python usando Wordnet Lemmatizer, Spacy Lemmatizer, TextBlob, Stanford CoreNLP.
# Importing Lemmatizer library from nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer() 
 
print("rocks :", lemmatizer.lemmatize("rocks")) 
print("corpora :", lemmatizer.lemmatize("corpora"))

############################################################################################################
#### Stop Words
############################################################################################################
# Son las palabras más comunes en un idioma como "el", "a", "en", "para", "sobre", "en", "es", "todo". 
# Estas palabras no tienen ningún significado y normalmente se eliminan de los textos. 
# Podemos eliminar estas palabras de parada usando la biblioteca nltk

# importing stopwors from nltk library
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
a = set(stopwords.words('english'))
text = "Cristiano Ronaldo was born on February 5, 1985, in Funchal, Madeira, Portugal."
text1 = word_tokenize(text.lower())
print(text1)
stopwords = [x for x in text1 if x not in a]
print(stopwords)

############################################################################################################
#### Part of speech tagging (POS)
############################################################################################################
# El etiquetado de partes de oración se utiliza para asignar partes de oración a cada palabra de un texto determinado 
# (como sustantivos, verbos, pronombres, adverbios, conjunción, adjetivos, interjección) basándose en su definición y su contexto. 
# Hay muchas herramientas disponibles para los etiquetadores POS y algunos de los etiquetadores más utilizados son NLTK, Spacy, 
# TextBlob, Standford CoreNLP, etc.
nltk.download('averaged_perceptron_tagger')
text = "vote to choose a particular man or a group (party) to represent them in parliament"
#Tokenize the text
tex = word_tokenize(text)
for token in tex:
    print(nltk.pos_tag([token]))

############################################################################################################
#### Named entity recognition
############################################################################################################
# Es el proceso de detectar las entidades nombradas como el nombre de la persona, el nombre de la ubicación, 
# el nombre de la empresa, las cantidades y el valor monetario.
text = "Google’s CEO Sundar Pichai introduced the new Pixel at Minnesota Roi Centre Event"
#importing chunk library from nltk
from nltk import ne_chunk
nltk.download('maxent_ne_chunker')
nltk.download('words')
# tokenize and POS Tagging before doing chunk
token = word_tokenize(text)
tags = nltk.pos_tag(token)
chunk = ne_chunk(tags)
print(chunk)

############################################################################################################
#### Chunking
############################################################################################################
# El "chunking" significa recoger trozos individuales de información y agruparlos en trozos más grandes. 
# En el contexto de la NLP y la minería de textos, "chunking" significa una agrupación de palabras o tokens en trozos.
text = "We saw the yellow dog"
token = word_tokenize(text)
tags = nltk.pos_tag(token)
reg = "NP: {<DT>?<JJ>*<NN>}"
a = nltk.RegexpParser(reg)
result = a.parse(tags)
print(result)