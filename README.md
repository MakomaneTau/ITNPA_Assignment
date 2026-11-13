# ITNPA Assignment

Research-abstract clustering and classification, extractive article summarisation,
and topic modelling for Questions 1–4.

## Submission contents (Question 4.3)

| Requirement | Repository location |
|---|---|
| Project datasets | [Research abstracts](Dataset/research_abstracts.json) and [article corpus](Dataset/research_corpus/) |
| Analytics flow for Questions 1–2 | [Question 1 notebook](Notebooks/Question%201/Notebook.ipynb), including Question 2 classification and tuning |
| Analytics flow for Questions 3–4 | [Question 3 notebook](Notebooks/Question%203/Notebook.ipynb), including Question 4 LDA and intertopic mapping |
| Deployment folder and instructions for Questions 1–2 | [Deployment guide](deployment/README.md) |

The two notebook filenames reflect their starting questions. Together they cover
all four questions; run each from top to bottom.

## Run the classifier

Use Python 3.13 (the locally checked interpreter is Python 3.13.6). From the
repository root, run these PowerShell commands:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Paste an abstract into the browser interface and select **Classify Abstract**.
The deployed predictor is the tuned Naive Bayes pipeline from Question 2. It
returns `ai` or `classical`, displayed as **Ai** or **Classical**. The interface
does not expose Question 1's exploratory clustering controls.

The first prediction may download NLTK data. See the
[deployment guide](deployment/README.md) for preparing those resources in advance,
rebuilding the saved model, and troubleshooting.

## Run the analysis notebooks

Install the analysis dependencies into the same environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Launch Question 1 from its own directory because its dataset path is relative to
the notebook working directory:

```powershell
Set-Location "Notebooks/Question 1"
..\..\.venv\Scripts\python.exe -m notebook Notebook.ipynb
```

Stop the notebook server with Ctrl+C when finished. From that directory, launch
the notebook covering Questions 3–4:

```powershell
Set-Location "../Question 3"
..\..\.venv\Scripts\python.exe -m notebook Notebook.ipynb
```

Select the environment's Python kernel and run all cells in order. In VS Code,
use the same interpreter and set the notebook working directory to the notebook's
folder. Initial package and NLTK installation requires internet access.


## Repository layout

```text
Dataset/                 Abstract JSON and five supplied article text files
Notebooks/Question 1/    Questions 1 and 2: clustering and classification
Notebooks/Question 3/    Questions 3 and 4: summaries, LDA
deployment/README.md     Classifier installation, model preparation and launch
src/best_model.py        Train, evaluate and save the tuned classifier
src/text_preprocessing.py Shared preprocessing and NLTK resource setup
model/model.pkl          Saved classifier pipeline
app.py                   Streamlit prediction interface
requirements.txt         Application and training dependencies
```

Dependencies are not locked; package upgrades can change results or prevent an
older pickle from loading. Rebuild the model in the serving environment when
needed, as described in the deployment guide. Notebook results describe a fixed
dataset and split, rather than guaranteed performance on new abstracts.

## License

See [LICENSE](LICENSE) for the repository's MIT license. Supplied article texts
retain their original authorship; the code license does not establish permission
to relicense third-party source material.
