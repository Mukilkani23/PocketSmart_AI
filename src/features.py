"""Text normalisation + TF-IDF featuriser for merchant strings.

Why CHARACTER n-grams (2-4) and not word n-grams:
  Bank merchant strings are not language. They are truncated ("SWIGGY  TRICH"),
  concatenated ("bundltechnol@upi"), misspelt ("SWIGY"), and glued to reference
  numbers and bank codes. Word tokenisation turns "ZOMATOONLINE" and "ZOMATO ONLINE"
  into unrelated tokens, and every truncation creates a brand-new, unseen word.
  Character 2-4-grams ("ZOM", "OMA", "MAT") survive truncation, concatenation and
  small typos because most of a string's n-grams are still shared. So the model
  generalises to spellings it never saw in training.

Why TF-IDF and not embeddings:
  ~7.7k short training strings, a CPU-only container, and a viva where every weight
  must be explainable. TF-IDF + a linear model is fast (milliseconds), deterministic,
  and its coefficients can be read directly ("'IGG' pushes toward Food & Dining").
  Sentence embeddings are trained on natural language and add a large dependency
  for strings that aren't natural language.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer

_DIGIT_RUN = re.compile(r"\d{4,}")
_SPACES = re.compile(r"\s+")


def mask_digits(s: str) -> str:
    """Replace runs of >=4 digits (phone numbers, UPI refs, account numbers) with '#'.

    Used for PRIVACY on real strings before they are written anywhere, and applied to
    every string (synthetic too) so training and real data share one representation.
    Digits carry no category signal. They are reference numbers.
    """
    return _DIGIT_RUN.sub("#", s)


def normalise(s: str) -> str:
    s = mask_digits(str(s)).upper()
    return _SPACES.sub(" ", s).strip()


def normalise_many(strings) -> list[str]:
    return [normalise(s) for s in strings]


def make_vectorizer() -> TfidfVectorizer:
    # char_wb pads each word with spaces, so n-grams mark word boundaries ("SWI" at a
    # word start differs from inside a word). sublinear_tf dampens repeated n-grams.
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        sublinear_tf=True,
        preprocessor=normalise,
        lowercase=False,
    )
