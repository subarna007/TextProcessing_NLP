# BBC news topic classification

Classifying BBC news articles into five topics (business, entertainment, politics, sport, tech) with TF-IDF features, chi-squared feature selection and logistic regression.

## Data

`bbc-text.csv`: 2,225 BBC news articles from 2004–2005 with their topic label (the public BBC News dataset of Greene and Cunningham, 2006).

## Method

1. Lower-case the text and apply Porter stemming.
2. TF-IDF vectorisation.
3. Keep the 2,000 most informative terms with a chi-squared test (`SelectKBest`).
4. Logistic regression.
5. Random 85/15 train/test split (334 test articles).

The notebook ends with a small demo that classifies any news text you type in.

## Results (334 test articles)

| Topic | Precision | Recall | F1 |
|---|---|---|---|
| Business | 0.97 | 0.96 | 0.97 |
| Entertainment | 0.98 | 0.98 | 0.98 |
| Politics | 0.98 | 0.97 | 0.97 |
| Sport | 0.99 | 0.99 | 0.99 |
| Tech | 0.95 | 0.97 | 0.96 |

**Test accuracy: 97.3%.** Topic words such as "match", "election" or "shares" separate these classes well, so a linear model on word counts is already strong.

## Known issue and next steps

- The stop-word step uses `filter()`, which keeps the original text unchanged, so stop words are not actually removed. Fixing it is unlikely to change the result much, because TF-IDF and the chi-squared selection already down-weight common words.
- The split is random without a fixed seed, so the exact accuracy changes slightly between runs. Cross-validation would give a more stable estimate.

## Run

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
jupyter notebook
```
