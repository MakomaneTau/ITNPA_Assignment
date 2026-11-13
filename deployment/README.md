# Deploy the Questions 1–2 classifier

This guide runs the Streamlit interface locally. Question 1 develops an exploratory
K-Means baseline; Question 2 compares supervised classifiers and tunes Naive Bayes.
The application serves that tuned classifier, accepting raw abstract text and
returning its predicted `ai` or `classical` label. The saved pipeline includes
preprocessing, TF-IDF and classification. No API key or database is required.

## 1. Obtain the project

Install Git and Python 3.13, then run in PowerShell:

```powershell
git clone https://github.com/MakomaneTau/ITNPA_Assignment.git
Set-Location ITNPA_Assignment
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you already have the repository, start in its root folder (the folder containing
`app.py`) and run the environment and installation commands there. These commands
use the environment's Python directly, so PowerShell activation is unnecessary.
Python 3.13.6 is the locally checked interpreter; this is not a locked environment.

## 2. Prepare the language resources

```powershell
.\.venv\Scripts\python.exe -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng wordnet
```

Run this as the same operating-system user that will run the application. The app
also attempts to download missing resources on the first prediction. Preparing
them beforehand allows later predictions without downloading language resources.

## 3. Prepare the model

The repository includes `model/model.pkl`. Try it first in the environment above.
Keep `src/text_preprocessing.py` with it: the pickle refers to its preprocessing
function by module name.

If the model is missing, incompatible with the installed scikit-learn version, or
you want to regenerate it with the current training script, run from the root:

```powershell
.\.venv\Scripts\python.exe -m src.best_model
```

This loads `Dataset/research_abstracts.json`, removes duplicate/missing records,
uses a fixed 80/20 split, and searches nine Naive Bayes/TF-IDF configurations with
five-fold stratified cross-validation on the training set. It prints held-out
metrics, writes **over** `model/model.pkl`, then checks that reloading preserves
test predictions. Keep a copy of the existing model first if you need that exact
artifact. Training uses all available job workers and can take several minutes.

Only load pickle files from a trusted source; loading a pickle can execute code.

## 4. Start the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed in the terminal (normally `http://localhost:8501`).
Paste a nonempty research abstract and select **Classify Abstract**. The page
shows **Ai** or **Classical**. Stop the server with Ctrl+C.

For a repeatable package record after verifying your own environment:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > deployment/installed-packages.txt
```

Keep that environment record with the corresponding model when transferring it
to another machine. The repository's requirements file is an unpinned dependency
list, not a guarantee of compatibility with every future package version.

## 5. Check the deployment

1. Confirm the page opens and the abstract input and classification button appear.
2. Submit an empty input and confirm the page asks for an abstract.
3. Submit a real abstract from the dataset and confirm a label appears without an
   error. A returned label verifies prediction plumbing; it does not prove the
   label is correct.
4. Try another abstract and confirm the app continues to respond.

The application reports a class label, not a calibrated confidence score or a
reliable assessment of whether an author used generative AI.

## Troubleshooting

| Symptom | Action |
|---|---|
| `No module named streamlit` or another missing package | Install `requirements.txt` using the same `.venv` Python used to launch the app. |
| NLTK lookup or download error | Run step 2 with internet access and a writable NLTK data directory under the serving user. |
| Model file not found | Run step 3 and check that `model/model.pkl` exists. |
| Pickle import/version error or inconsistent-version warning | Preserve the original artifact if needed, rebuild with step 3, then restart the app. |
| `No module named src` | Launch from the repository root and retain the `src` directory. |
| Port 8501 already in use | Append `--server.port 8502` to the Streamlit launch command. |

This guide establishes a local deployment. A public hosted URL is a separate
deployment and is not supplied by these instructions.
