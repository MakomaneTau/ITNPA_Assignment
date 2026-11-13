"""Run Question 4.2: build an offline interactive LDA map and open a browser.

Usage: python intertopic_map.py
       python intertopic_map.py --no-browser --output my_map.html

Preprocessing and LDA settings match the Question 3 notebook (Questions 4.1?4.2).
"""

import argparse
import hashlib
import re
import unicodedata
import webbrowser
from pathlib import Path

import nltk
from nltk import pos_tag, sent_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pyLDAvis
import pyLDAvis.lda_model
from pyLDAvis import urls as ldavis_urls
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_DIR = Path(__file__).resolve().parent
lemmatizer = WordNetLemmatizer()


def ensure_resources():
    """Download missing NLTK data once, before preprocessing articles."""
    for resource, location in (
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("wordnet", "corpora/wordnet"),
    ):
        try:
            nltk.data.find(location)
        except LookupError:
            try:
                nltk.data.find(location + ".zip")
            except LookupError:
                if not nltk.download(resource, quiet=True, raise_on_error=True):
                    raise RuntimeError(f"Could not download NLTK resource: {resource}")


def sentence_key(text):
    return ' '.join(re.findall(r"[a-z]+", text.lower()))

def denoise_article(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.translate(str.maketrans({
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': ' - ', '\u2014': ' - ', '\ufeff': '', '\u03b1': ' alpha ',
    }))
    text = ''.join(
        char if not unicodedata.category(char).startswith('C') or char in '\n\r\t' else ' '
        for char in text
    )
    text = re.sub(r'(?<=[a-z])-\s*\n\s*(?=[a-z])', '', text)
    lines = text.splitlines()
    body_start = 0
    for index, line in enumerate(lines[:30]):
        if re.fullmatch(r'\s*(?:abstract|(?:\d+(?:\.\d+)*\s+)?introduction)\s*', line, re.I):
            body_start = index + 1
            break

    kept_lines = []
    in_table = False
    in_footnote = False
    skip_next = False
    removed_lines = body_start
    for line_number, line in enumerate(lines[body_start:], start=body_start):
        line = line.strip()
        if skip_next:
            skip_next = False
            removed_lines += 1
            continue
        if not line:
            continue
        if in_footnote or re.match(r'^[1-9]\d?(?=[A-Z])', line):
            in_footnote = not bool(re.search(r'[.!?][\"\']?$', line))
            removed_lines += 1
            continue
        if re.fullmatch(r'(?:references|bibliography|works cited)', line, re.I):
            removed_lines += len(lines) - line_number
            break
        if re.match(r'^(?:corresponding author|keywords)\s*:?$', line, re.I):
            skip_next = True
            removed_lines += 1
            continue
        if (re.match(r'^(?:TABLE|FIGURE)\s+\d', line)
                or re.fullmatch(r'(?:table|figure)\s+\d+(?:\.\d+)*\.?', line, re.I)):
            in_table = True
            removed_lines += 1
            continue
        # Tables in these text exports contain short fragments; resume at prose.
        if in_table:
            is_prose = (len(line.split()) >= 12 and re.search(r'[.!?](?:\s|$)', line))
            if is_prose or (re.match(r'^[a-z]', line) and len(line.split()) >= 8 and '&' not in line):
                in_table = False
            else:
                removed_lines += 1
                continue
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", line)
        is_title = (2 <= len(words) <= 16
                    and sum(word[0].isupper() for word in words) / len(words) >= 0.65
                    and not re.search(r'[.!?;]$', line))
        is_header = (
            is_title
            or
            re.fullmatch(r'\d{1,4}', line)
            or re.fullmatch(r'[A-Z][a-z]+\s+\d{2,4}', line)
            or re.match(r'^\d+\s+Qualitative Research\b', line)
            or (line.isupper() and len(line.split()) <= 14)
            or re.match(r'^\d+(?:\.\d+)+\s+[A-Z]', line)
            or re.match(r'^\d+\.\s+DEVELOPMENTAL RESEARCH\b', line)
            or re.match(r'^(?:by |Email:|https?://|doi:|Copyright|Article$|Full size image$)', line, re.I)
            or re.match(r'^\d{6,}\w*10\.\d{4,}/', line)
        )
        if is_header:
            removed_lines += 1
            continue
        kept_lines.append(line)

    body = re.sub(r'https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.\w+', ' ', ' '.join(kept_lines))
    # Remove parenthetical citations containing a publication year, not all parentheses.
    for _ in range(2):
        body = re.sub(r'\([^()]*\b(?:18|19|20)\d{2}[a-z]?\b[^()]*\)', ' ', body)
    body = re.sub(r'(?<=[A-Za-z][.!?])\d{1,2}(?:\s*,\s*\d{1,2})*\b', '', body)
    body = re.sub(r'\(p{1,2}\.\s*\d+(?:[-,]\s*\d+)*\)', ' ', body)
    body = re.sub(r'(?<=[a-z])\d{1,2}(?=\W|$)', '', body)
    body = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', body)
    body = re.sub(r'(?<=[.!?])(?=[A-Z])', ' ', body)
    body = re.sub(r'\s+([,.;:!?])', r'\1', body)
    body = re.sub(r'\s+', ' ', body).strip()

    sentences = []
    seen = set()
    repeated_sentences = 0
    for sentence in sent_tokenize(body):
        sentence = sentence.strip()
        key = sentence_key(sentence)
        if len(key.split()) < 5:
            continue
        if key in seen:
            repeated_sentences += 1
            continue
        seen.add(key)
        sentences.append(sentence)
    return sentences, removed_lines, repeated_sentences

def lemmatize_text(text):
    words = re.findall(r'[^\W\d_]+', text.lower(), flags=re.UNICODE)
    lemmas = []
    for word, tag in pos_tag(words):
        position = {'J': wordnet.ADJ, 'V': wordnet.VERB,
                    'N': wordnet.NOUN, 'R': wordnet.ADV}.get(tag[:1], wordnet.NOUN)
        lemmas.append(lemmatizer.lemmatize(word, position))
    return ' '.join(lemmas)


def build_map(corpus_dir):
    """Fit the notebook's four-topic LDA model to complete supplied articles."""
    files = sorted(corpus_dir.glob("*.txt"))
    if len(files) < 2:
        raise ValueError(f"At least two article .txt files are required in {corpus_dir}")
    ensure_resources()
    documents = []
    hashes = set()
    for file in files:
        raw = file.read_bytes()
        hashes.add(hashlib.sha256(raw).hexdigest())
        sentences, _, _ = denoise_article(raw.decode("utf-8-sig"))
        text = " ".join(lemmatize_text(sentence) for sentence in sentences)
        if not text.strip():
            raise ValueError(f"No usable prose remains after cleaning {file.name}")
        documents.append(text)
    print(f"Loaded {len(files)} articles ({len(hashes)} distinct contents).")
    if len(hashes) < len(files):
        print("Duplicate articles are retained, matching the assignment notebook.")
    vectorizer = CountVectorizer(stop_words="english", max_df=0.95)
    matrix = vectorizer.fit_transform(documents)
    if (matrix.sum(axis=1) == 0).any():
        raise ValueError("An article has no retained terms; check the supplied text files.")
    model = LatentDirichletAllocation(
        n_components=4, random_state=42, learning_method="batch"
    )
    model.fit(matrix)
    return pyLDAvis.lda_model.prepare(
        model, matrix, vectorizer,
        mds="tsne", plot_opts={"xlab": "t-SNE 1", "ylab": "t-SNE 2"},
        sort_topics=False, n_jobs=1,
    )


def standalone_html(prepared):
    """Embed installed JavaScript/CSS assets so the saved map works offline."""
    page = pyLDAvis.prepared_data_to_html(
        prepared, template_type="simple", visid="research-lda-map",
        d3_url="d3.js", ldavis_url="ldavis.js", ldavis_css_url="ldavis.css",
    )
    for name, path in (("d3.js", ldavis_urls.D3_LOCAL),
                       ("ldavis.js", ldavis_urls.LDAVIS_LOCAL)):
        tag = f'<script type="text/javascript" src="{name}"></script>'
        if tag not in page:
            raise RuntimeError(f"Cannot embed {name}: unexpected pyLDAvis HTML format")
        script = Path(path).read_text(encoding="utf-8").replace("</script", r"<\/script")
        page = page.replace(tag, f'<script type="text/javascript">{script}</script>')
    tag = '<link rel="stylesheet" type="text/css" href="ldavis.css">'
    if tag not in page:
        raise RuntimeError("Cannot embed CSS: unexpected pyLDAvis HTML format")
    page = page.replace(tag, "<style>" + Path(ldavis_urls.LDAVIS_CSS_LOCAL).read_text(
        encoding="utf-8") + "</style>")
    if re.search(r'<script[^>]+src=|<link[^>]+href=', page, re.I):
        raise RuntimeError("The generated map still references external assets")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Question 4.2 ? Interactive topic map</title>'
        '<style>body{margin:24px;font-family:Arial,sans-serif;background:white;color:#111;}'
        'main{max-width:1300px;margin:auto;}p{line-height:1.5;max-width:1000px;}'
        '.map{overflow-x:auto;}#research-lda-map{min-width:1210px;}'
        'svg text{fill:#111;}button,input{color:#111;}</style></head><body><main>'
        '<h1>Intertopic distance map</h1>'
        '<p>Select a topic circle or use <b>Next Topic</b> to explore its terms. '
        'Move the relevance slider (?) to compare common words with words distinctive '
        'to that topic. Clear the topic selection to see the most salient terms '
        'across the corpus. Hover over a word to explore its topic associations.</p>'
        '<p>Circle area shows token-weighted topic prevalence. Topic numbers match '
        'Question 4.1. Distances are a qualitative two-dimensional t-SNE projection.</p>'
        '<p>This small corpus includes duplicate articles. A negligible topic and '
        'merged themes limit interpretation; four configured topics do not establish '
        'four distinct research trends.</p><div class="map">'
        + page + '</div></main></body></html>'
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path,
                        default=PROJECT_DIR / "Dataset" / "research_corpus",
                        help="Folder containing the article .txt files")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_DIR / "deployment" / "lda_intertopic_map.html",
                        help="Destination for the standalone interactive HTML map")
    parser.add_argument("--no-browser", action="store_true",
                        help="Save the map without opening a browser")
    args = parser.parse_args(argv)
    try:
        print("Preparing articles and fitting four LDA topics...")
        prepared = build_map(args.corpus)
        page = standalone_html(prepared)
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(page, encoding="utf-8")
    except (OSError, ValueError, LookupError, RuntimeError) as error:
        parser.exit(1, f"Could not create the topic map: {error}\n")
    print(f"Interactive map saved to: {output}")
    if not args.no_browser:
        if not webbrowser.open(output.as_uri()):
            print("Open the saved HTML file in a web browser to interact with it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
