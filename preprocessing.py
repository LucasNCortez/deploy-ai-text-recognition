import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import unicodedata
import spacy
import os

os.system('pip install models/en_core_web_sm-3.8.0-py3-none-any.whl')

nlp_spacy = spacy.load('en_core_web_sm')

def remove_excessive_spaces(text: str) -> str:
    """
    This function removes excessive spaces from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with excessive spaces removed.
    """
    return re.sub(r'\s+', ' ', text).strip() 

def remove_repeated_non_word_characters(text: str) -> str:
    """
    This function removes repeated non-word characters from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with repeated non-word characters removed.
    """
    return re.sub(r'(\W)\1+', r'\1', text).strip()

def remove_first_line_from_text(text: str) -> str:
    """
    This function removes the first line from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with the first line removed.
    """
    return re.sub(r'^.*\n', '', text).strip()

def remove_last_line_from_text(text: str) -> str:
    """
    This function removes the last line from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with the last line removed.
    """
    return re.sub(r'\n.*$', '', text).strip()

def fix_isolated_commas_in_text(text: str) -> str:
    """
    This function fixes isolated commas in the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with isolated commas fixed.
    """
    return re.sub(r' ([.,:;!?])', r'\1', text).strip()

def keep_words_longer_than(text: str, min_length: int = 2) -> str:
    """
    This function keeps only the words in the text that are longer than a given length.

    Args:
        text (str): The input text.
        min_length (int, optional): The minimum length of the words to keep. Defaults to 2.

    Returns:
        str: The text with only the words longer than the given length.
    """
    return ' '.join([word for word in text.split() if len(word) > min_length])

def keep_only_alphabet_characters(text: str) -> str:
    """
    This function keeps only the alphabet characters in the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with only the alphabet characters.
    """
    return re.sub(r'[^a-zA-Z]', ' ', text).strip()

def remove_accents_from_text(text: str) -> str:
    """
    This function removes accents from the text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with accents removed.
    """
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

def lemmatize_text_with_spacy(text: str) -> str:
    """
    This function lemmatizes the text using the Spacy library.

    Args:
        text (str): The input text.

    Returns:
        str: The lemmatized text.
    """
    doc = nlp_spacy(text)
    return ' '.join([token.lemma_ for token in doc])

pipeline_clean_text = Pipeline([
    ('remove_first_line_from_text', FunctionTransformer(remove_first_line_from_text)),
    ('remove_last_line_from_text', FunctionTransformer(remove_last_line_from_text)),
    ('remove_excessive_spaces', FunctionTransformer(remove_excessive_spaces)),
    ('remove_repeated_non_word_characters', FunctionTransformer(remove_repeated_non_word_characters)),
    ('fix_isolated_commas_in_text', FunctionTransformer(fix_isolated_commas_in_text)),
    ('keep_only_alphabet_characters', FunctionTransformer(keep_only_alphabet_characters)),
    ('remove_accents_from_text', FunctionTransformer(remove_accents_from_text)),
    ('lemmatize_text_with_spacy', FunctionTransformer(lemmatize_text_with_spacy)),
])

def clean_text(text: str) -> str:
    return pipeline_clean_text.fit_transform([text])[0]
