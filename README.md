# Which Steam Hits Keep Their Players?

INSY 674 (Enterprise Data Science), McGill, Fall 2026: final group project
**Team:** Group 3

| Member | GitHub ID |
|---|---|
| Montserrat Balleza Velazquez | dev-Montsebllza |
| Mohamed Guessous | guessous-mohamed |
| Ishaan Khanna | ishaankhanna1998 |
| Miles Koseleci-Vieira | mileskv2 |
| Ruilei Ma | rl-ma |
| Daniel Puente Cavazos | cudapuca0 |
| Irene Wang | icecomfort |

Among Steam games that became hits (500+ reviews, released 2019–2023), we predict which ones still have **more players than expected for their size and age** years after launch, using only information available on launch day, and identify what drives staying power.

## Repository contents

| File | What it is |
|---|---|
| `steam_staying_power.ipynb` | The full project notebook (all five checklist sections, with outputs) |
| `README.md` | This file: summary of each section |
| `Steam_Staying_Power.pptx` | Presentation slides |

## How to run

1. Open `steam_staying_power.ipynb` in Google Colab or Jupyter.
2. *Run all*. The first cell downloads the two data files (about 190 MB parquet + 400 MB CSV) from Hugging Face.
3. A full run takes about 5 minutes plus download time.

Libraries: pandas, numpy, scikit-learn, matplotlib (all preinstalled in Colab).

**Data:** FronkonGames, *Steam Games Dataset*, https://huggingface.co/datasets/FronkonGames/steam-games-dataset. We use the version at Hugging Face commit `ba4e26785af33bee500e96597068c6e77f4edea9` (last updated 2 October 2026); the notebook downloads exactly this version. The parquet file is the main data; the CSV adds `Discount` (to recover regular prices) and player `Tags` (used only to remove idle games). The raw data isn't committed because it exceeds GitHub's file-size limit.

## 1. Frame the problem
- **Objective:** help developers and publishers understand which launch-day characteristics predict that a successful game keeps its players. That informs design choices and decisions about funding updates, DLC and sequels.
- **Framing:** supervised binary classification, offline (batch) learning.
- **Target, "lasting":** more current players than expected for a game of the same size (total reviews) and release year, in the top 25% of its year. Expected players come from a linear regression fitted on the training set only.
- **Metric:** PR-AUC (random guessing = 0.25), plus ROC-AUC, F1, precision and recall.

## 2. Explore the data
- About two-thirds of games were on sale on the snapshot day, so the price is corrected to the regular price.
- Bigger hits keep proportionally more players (10× the reviews → about 17× the current players). A simple "players per review" target was biased toward big games, so we size-adjusted it.
- Replayable genres (sports, strategy, MMO, simulation) last; action and adventure don't. Online and community features (multiplayer, Workshop, level editors) go with lasting games.
- Accessibility labels added to store pages years after launch are leakage.

## 3. Prepare the data
- **Leakage audit:** dropped all post-launch columns (reviews, owners, playtime, scores, DLC count, tags, discount) and leaky category values.
- **Feature engineering:** regular price, counts (languages, screenshots, achievements, social features), description lengths, release timing, sequel flag, multi-hot genres and categories, developer/publisher history from earlier games only.
- **Pipeline:** a single scikit-learn `ColumnTransformer`, fitted on training data only. 80/20 split stratified by release year.

## 4. Measure and compare models
9 model families compared with 5-fold CV. The best defaults were random forest (PR-AUC 0.588), logistic regression (0.565) and gradient boosting (0.562), which made up the shortlist. Feature selection dropped 7 uninformative features (including release year, as the target design intended).

## 5. Fine-tune the system
- **Tuning:** randomized search with preparation choices tuned as hyperparameters.
- **Ensemble:** soft voting of the three tuned models (CV PR-AUC 0.603).
- **Test set:** PR-AUC 0.578 vs. 0.258 for random guessing and 0.552 for an untuned logistic regression. 48% of the games flagged as lasting really were (1.9× the base rate).
- **Top drivers:** genre, regular price, store categories, developer experience.
- **Robustness:** removing the language features changes nothing; an alternative "50+ players" target gives the same main conclusions.

## Assumptions and limitations
See Section 7 of the notebook: one-day player counts, reviews as a size proxy, store features that can change after launch, only hits studied, and no post-launch update or community data.
