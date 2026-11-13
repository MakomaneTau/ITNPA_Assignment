import re

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def ensure_nltk_resources():
    for resource, location in (
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("wordnet", "corpora/wordnet.zip"),
    ):
        try:
            nltk.data.find(location)
        except LookupError:
            nltk.download(resource, raise_on_error=True)


def preprocess_abstracts(abstracts):
    lemmatizer = WordNetLemmatizer()
    cleaned = []

    for abstract in abstracts:
        text = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", abstract)).strip().lower()
        lemmas = []
        for word, tag in pos_tag(word_tokenize(text)):
            if tag.startswith("J"):
                pos = wordnet.ADJ
            elif tag.startswith("V"):
                pos = wordnet.VERB
            elif tag.startswith("R"):
                pos = wordnet.ADV
            else:
                pos = wordnet.NOUN
            lemma = lemmatizer.lemmatize(word, pos)
            if lemma.lower() not in ENGLISH_STOP_WORDS:
                lemmas.append(lemma)
        cleaned.append(" ".join(lemmas))

    return cleaned
