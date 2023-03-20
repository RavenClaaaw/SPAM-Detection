import numpy as np
import pandas as pd
import nltk # NATURAL LANGUAGE TOOLKIT
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from spellchecker import SpellChecker
import nltk
from nltk.stem import WordNetLemmatizer 
nltk.download('stopwords')
nltk.download("punkt")
nltk.download('omw-1.4')
nltk.download("wordnet")

punctuations = [i for i in string.punctuation]
stop_words = set((stopwords.words("english") + punctuations))

def PProcess(text):
  # Tokenize & Stop Word Removal:
  text = str.lower(text)
  word_tokens = word_tokenize(text)

  filtered = ""
  for w in word_tokens:
      if w.lower() not in stop_words:
          filtered += w 
          filtered += " "

  text = filtered

  correct = ""
  spell = SpellChecker()
  mistake = spell.unknown(text.split())

  for word in text.split():
      if word == None:
        continue
      if word in mistake:
          correct += str(spell.correction(word)) + " "
      else:
          correct += str(word) + " "

  text = correct

  LTZ = WordNetLemmatizer()

  word_tokens = word_tokenize(text)
  filtered = ' '.join([LTZ.lemmatize(word) for word in word_tokens])
  
  text = filtered
  
  return text

print(PProcess("Hello My Name Is Roop Kumar, And I Am Software Engineer!"))