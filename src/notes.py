"""Sorting the supervisors' notes into reasons for the extra hours.

No language model. There are 2,117 notes but only 846 distinct ones, and they are
roughly twenty templates permuted by surname, five spellings of 6am, abbreviations,
typos, and code-switching into Afrikaans and isiZulu. A closed vocabulary that small
is better served by a lexicon that is deterministic, instant, free and auditable.

Two independent methods are implemented so the sorting can be checked against
something other than itself: a keyword lexicon, and a character n-gram model that
never sees the lexicon.
"""

from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd

CATEGORIES = [
    "relief_no_show", "absence_cover", "late_handover",
    "equipment_failure", "client_requested", "client_requested_unapproved",
    "no_information",
]
OPERATIONAL = {"relief_no_show", "absence_cover", "late_handover", "equipment_failure"}
BILLABLE = {"client_requested"}

# Expansions found by reading the notes: abbreviations, and Afrikaans/isiZulu stems
# mapped onto the English concept they carry.
EXPANSIONS = {
    r"\bhrs\b": "hours", r"\bagn\b": "again", r"\bagin\b": "again", r"\bmgr\b": "manager",
    r"\bmgmt\b": "management", r"\bbc\b": "because", r"\bpls\b": "please",
    r"\bokd'?\b": "approved", r"\bob book\b": "handover book",
    r"conntrol": "control", r"\bclint\b": "client", r"cnetre": "centre",
    # Afrikaans
    r"\bklient\b": "client", r"stukkend": "broken", r"oorhandiging": "handover",
    r"gedek vir": "covered for", r"siek gemeld": "booked off sick", r"masjien": "machine",
    r"\baflos\b": "relief", r"niks om te rapporteer": "nothing to report",
    r"het nie opgedaag nie": "did not arrive", r"ekstra ure": "extra hours",
    r"gewag vir": "waited for", r"met die hand": "by hand",
    # isiZulu
    r"akafikanga": "did not arrive", r"akezanga": "did not arrive",
    r"akukho lutho": "nothing to report", r"ngicela": "please",
    r"ngimele": "covered for", r"ngihlale": "stayed on",
}

# Checked in order. Operational causes are deliberately tested before client
# requests: several notes read "client signed for the extra hrs but real reason is
# relief no show again", and a first-match rule would file the failure as billable.
PATTERNS: list[tuple[str, str]] = [
    ("relief_no_show",
     r"relief|did not pitch|didnt pitch|no show|nobody came|no replacement|next shift|"
     r"suppose to come|no one came|never pitched|stayed on"),
    ("absence_cover",
     r"cover(ing|ed)?\b|stood in for|didnt come in|did not come in|absent|booked off sick|"
     r"double duty|2 posts|two posts|at the clinic|took .* shift as well|rounds as well|"
     r"family responsibility"),
    ("equipment_failure",
     r"machine|scrubber|buffer|generator|lift|gate motor|broke|broken|kaput|out of order|"
     r"fault|down again|by hand|manually"),
    ("late_handover",
     r"handover|hand over|handover book|keys missing|waiting on paperwork|delayed by"),
    ("client_requested",
     r"client|centre manage|centre manager|centre management|site manager|approved|"
     r"signed off|signed for|per client|stocktake|event|deep clean|extra patrol|load in"),
]

NOTHING = re.compile(
    r"^(ntr|ok|okay|fine|sharp|n/?a|-+|\.+|all good|all quiet|quiet shift|no incidents|"
    r"nothing to report|as per normal|no issues on site|all fine|nothing|normal shift)$")

# A contrastive marker means the clause after it carries the real cause.
CONTRAST = re.compile(r"\b(but|real reason|maar|however)\b")
# Client asked, but nobody signed it off - possibly not billable at all.
UNAPPROVED = re.compile(
    r"d[o]{1,3}n'?t know|do not know|not sure|unsure|no approval|without approval|"
    r"weet nie of|angazi")


def normalise(text: str) -> str:
    """Fold away the noise so matching sees the sentence, not the typing."""
    out = str(text or "").lower().strip()
    out = re.sub(r"[.!]+$", "", out)
    out = re.sub(r"\b(0?6[:h]?00|6\s?am|six|0600|06h00)\b", "sixam", out)
    out = re.sub(r"\b\d{1,2}[:h]\d{2}\b", "attime", out)
    for pattern, replacement in EXPANSIONS.items():
        out = re.sub(pattern, replacement, out)
    return re.sub(r"\s+", " ", out).strip()


def classify_lexicon(text: str) -> str:
    """Method A: ordered keyword rules over the normalised text."""
    clean = normalise(text)
    if not clean or NOTHING.match(clean):
        return "no_information"

    # If the note contrasts itself, only the clause after the marker is trusted.
    match = CONTRAST.search(clean)
    decisive = clean[match.end():] if match else clean

    for name, pattern in PATTERNS:
        if re.search(pattern, decisive):
            if name == "client_requested" and UNAPPROVED.search(clean):
                return "client_requested_unapproved"
            return name

    if match:                                   # nothing after the marker; try the whole note
        for name, pattern in PATTERNS:
            if re.search(pattern, clean):
                return name
    return "no_information"


def classify_vectors(texts: list[str], seed_rows: list[int],
                     seed_labels: list[str]) -> list[str]:
    """Method B: character n-grams and nearest centroid.

    Deliberately independent of the lexicon's vocabulary: it sees only characters,
    so it survives typos and code-switching without being told about either. Seeds
    are the notes the lexicon is most confident about, which makes this a test of
    whether the categories are real structure in the text rather than an opinion.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    cleaned = [normalise(t) for t in texts]
    vectoriser = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 sublinear_tf=True)
    normed = normalize(vectoriser.fit_transform(cleaned)).tocsr()

    labels = sorted(set(seed_labels))
    centroids = []
    for label in labels:
        rows = [seed_rows[i] for i, s in enumerate(seed_labels) if s == label]
        centroids.append(np.asarray(normed[rows].mean(axis=0)).ravel())
    centroids = np.vstack(centroids)
    centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)

    scores = np.asarray(normed @ centroids.T)
    return [labels[i] for i in scores.argmax(axis=1)]


def classify_all(conn: sqlite3.Connection) -> pd.DataFrame:
    """Every note, both methods, side by side."""
    notes = pd.read_sql("SELECT shift_id, logged_by, note FROM shift_notes", conn)
    notes["note"] = notes.note.fillna("")
    notes["normalised"] = notes.note.map(normalise)
    notes["lexicon"] = notes.note.map(classify_lexicon)

    # Seed the vector model on the templates the lexicon sees most often, so it is
    # learning the shape of each category rather than memorising rare one-offs.
    common = notes.normalised.value_counts()
    seeds = notes[notes.normalised.map(common).ge(3)]
    notes["vectors"] = (
        classify_vectors(notes.note.tolist(), seeds.index.tolist(), seeds.lexicon.tolist())
        if len(seeds) else notes.lexicon)
    notes["agree"] = notes.lexicon == notes.vectors

    # Production label. The lexicon decides what it recognises, because its ordering
    # rules encode something a bag of words cannot; Naive Bayes picks up the rest,
    # and abstains rather than guessing when it recognises nothing at all.
    model = train_naive_bayes(conn)
    notes["naive_bayes"] = model.predict(notes.note.tolist())
    notes["category"] = [classify_hybrid(t, model) for t in notes.note]
    return notes


def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    """Agreement beyond what two methods would hit by chance alone."""
    observed = (a == b).mean()
    labels = sorted(set(a) | set(b))
    expected = sum((a == l).mean() * (b == l).mean() for l in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


# --------------------------------------------------------------------------- #
# Method C: bag of words + Naive Bayes
# --------------------------------------------------------------------------- #
#
# The lexicon is precise but brittle: it only knows the words I thought to write
# down. Naive Bayes learns which words actually go with which cause, from the
# whole corpus, so it can recognise a phrasing nobody anticipated as long as the
# words themselves have been seen.
#
# It is trained on the lexicon's own labels (distant supervision), which means it
# cannot be judged by agreeing with the lexicon - that is circular. The real test
# is held-out notes written in language the lexicon was never given, and that is
# what labels/handcrafted_notes.csv is for.

# scikit-learn's default tokenizer does the splitting - the standard
# "(?u)\b\w\w+\b" word pattern with English stop words and unigrams plus bigrams.
# Pairs are worth the extra features here: "no show", "did not" and "signed off"
# each mean something their individual words do not.
#
# The only custom part is the preprocessor, which is `normalise` - it folds the
# five spellings of 6am together, expands the abbreviations, and maps the Afrikaans
# and isiZulu stems onto the English concept they carry. Without that step every
# language would need its own vocabulary learned separately from very few examples.


class NaiveBayesNotes:
    """Multinomial Naive Bayes over a bag of words.

    Kept as a thin wrapper so production code does not care which method is in use.
    """

    #: Add-one (Laplace) smoothing. Every word gets a pretend extra count in every
    #: category, so a word never seen alongside a category does not drive its
    #: probability to zero and veto the whole note on one unfamiliar term.
    def __init__(self, alpha: float = 1.0, min_df: int = 1) -> None:
        self.alpha, self.min_df = alpha, min_df
        self.pipeline = None

    def fit(self, texts: list[str], labels: list[str]) -> "NaiveBayesNotes":
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import make_pipeline

        self.pipeline = make_pipeline(
            CountVectorizer(preprocessor=normalise, ngram_range=(1, 2),
                            stop_words="english", min_df=self.min_df),
            MultinomialNB(alpha=self.alpha),
        )
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        """Predict, but abstain when there is nothing to go on.

        Naive Bayes always answers. If a note contains no word the model has ever
        seen, every category is equally uninformed and it quietly returns whichever
        class was largest in training - so "zzz" comes back as absence cover, stated
        with the same confidence as a real prediction. Abstaining is the honest
        answer: no evidence is not the same as weak evidence.
        """
        vectoriser = self.pipeline.named_steps["countvectorizer"]
        known = np.asarray(vectoriser.transform(texts).sum(axis=1)).ravel()
        predicted = self.pipeline.predict(texts)
        return ["no_information" if n == 0 else p for n, p in zip(known, predicted)]

    def predict_proba(self, texts: list[str]):
        return self.pipeline.predict_proba(texts)

    @property
    def classes_(self):
        return self.pipeline.classes_


def train_naive_bayes(conn: sqlite3.Connection) -> NaiveBayesNotes:
    """Fit on every note in the database, labelled by the lexicon."""
    notes = pd.read_sql("SELECT note FROM shift_notes", conn).note.fillna("").tolist()
    return NaiveBayesNotes().fit(notes, [classify_lexicon(n) for n in notes])


def classify_hybrid(text: str, model: NaiveBayesNotes) -> str:
    """What production uses.

    The lexicon answers when it recognises something, because its rules encode
    ordering that a bag of words cannot represent - notably that in "client signed
    the hours but the real reason was the relief", the clause after "but" is the
    one that counts. Naive Bayes answers everything the lexicon does not recognise,
    which is where unfamiliar phrasing lands.
    """
    clean = normalise(text)
    if not clean or NOTHING.match(clean):
        return "no_information"

    # A note that contradicts itself - "client signed the hours but the real reason
    # was the relief" - is decided by the clause after the marker, and by that
    # clause alone. Reading the whole note lets the client keywords win, which is
    # precisely the mistake that moves a preventable failure into the billable pile.
    marker = CONTRAST.search(clean)
    if marker:
        decisive = clean[marker.end():].strip()
        if decisive:
            verdict = classify_lexicon(decisive)
            return verdict if verdict != "no_information" else model.predict([decisive])[0]

    verdict = classify_lexicon(text)
    return verdict if verdict != "no_information" else model.predict([text])[0]


def classify_all_methods(conn: sqlite3.Connection, texts: list[str],
                         model: "NaiveBayesNotes | None" = None) -> pd.DataFrame:
    """Every method side by side, for evaluation."""
    model = model or train_naive_bayes(conn)
    return pd.DataFrame({
        "note": texts,
        "lexicon": [classify_lexicon(t) for t in texts],
        "naive_bayes": model.predict(list(texts)),
        "hybrid": [classify_hybrid(t, model) for t in texts],
    })
