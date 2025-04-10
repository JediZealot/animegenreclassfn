# Anime Genre Classification with SBERT + TF-IDF + Neural Networks

Welcome to the land of waifus, warriors, and wicked cool deep learning! This repo tackles the problem of multi-label genre classification for anime based on their plot synopsis using two NLP techniques: SBERT embeddings and TF-IDF vectors.

## Problem Statement

Anime often spans multiple genres (like horror-romance-comedy-action-what-is-even-going-on), so this isn’t just a vanilla classification problem. It's multi-label, which means each anime can belong to multiple genres at once.

## Dataset

Anime dataset was collected from MyAnimeList using Jikan API

Final dataset `anime_dataset_preprocessed.csv`
Contains:
- `anime_id`
- `title`
- `synopsis`
- `combined` (Multilabel Genres)

## Data Preprocessing

The data preprocessing pipeline is structured in two key stages to clean, filter, and prepare the anime dataset before it goes to the model.

### Step 1: `anime_clean.ipynb` – Data Cleaning & Filtering

This notebook lays the foundation by preparing the dataset structurally and semantically:

1. Column Selection & Combination

     - Retained only relevant columns: `mal_id`, `title`, `synopsis`, `genres`, `themes`.

      - Merged `genres` and `themes` into a single combined column for genre labeling.

      - Dropped the original individual `genre` columns for a cleaner structure.

2. Handling Missing Values

      - Identified and removed rows with missing or empty `synopsis`.

3. Filtering Uninformative Data

      - Removed rows with short or generic synopses (less than 20 words) and clean source attribution (e.g., "[Written by MAL Rewrite]").

      - Eliminated entries likely to be specials, OVAs, sequels, recaps, or movies, using regex filtering on keywords like "OVA", "DVD", "movie", "recap", etc.

      - Used a curated list of mal_ids to manually retain a set of specific entries (e.g., to keep entries with informative synopses).

4. Genre Standardization
      - Only certain genres were kept due to some genres not having enough data (e.g., Showbiz - 39 animes, Love Status Quo - 37 animes)
      - The genre/themes were standardized to a controlled vocabulary of 16 main categories [ "Action", "Adventure", "Comedy", "Drama", "Ecchi", 
                      "Fantasy", "Historical", 
                      "Mecha", "Music", "Mystery", 
                      "Romance", "School", "Sci-Fi", "Slice of Life", "Supernatural"]
   - Some genre names were simplified for clarity. (e.g., "Ecchi" = "Lewd", "Sci-Fi": "Science Fiction", etc.)


5. Duplicate Handling
   - Multiple steps were taken to identify and remove duplicate entries:

        1. Exact duplicate rows
        2. Duplicate synopses with different titles
        3. Near-duplicate entries

6. Saved Output

      - Result saved to `anime_dataset_cleaned.csv` — 10,708 unique anime entries, clean, relevant, and ready for text processing.

### Step 2: `preprocessing.ipynb` – Text Normalization
Once we’ve got high-quality, relevant anime entries, this stage takes over to prep the text itself:


1. Text Cleaning

   - Lowercased all text.

   - Removed digits and punctuation.

   - Tokenized sentences using NLTK.

2. Stopword Removal & Lemmatization

   - Removed common English stopwords.

   - Applied lemmatization with WordNetLemmatizer.

3. Progress Visualization

   - Used tqdm to track preprocessing progress across ~thousands of rows.

4. Final Output

    - The cleaned dataset, with normalized synopsis text, is saved as `anime_dataset_preprocessed.csv`

##  Feature Extraction
Two vectorization strategies are used:
1. SBERT Embeddings

    - Model: `all-mpnet-base-v2`

    - Provides semantic embeddings for each synopsis.

2. TF-IDF Vectorization

    - Max features: 5000

    - Stopwords removed (english)

    - More traditional bag-of-words vibe.

## Model Architecture
- Multilabel Stratified K-Fold to ensure that each fold maintains the same proportion of genre labels, preventing genre imbalance during training and evaluation — crucial for multi-label problems where some genres are rare.

- A simple feed-forward neural network (built with PyTorch).

- Activation: ReLU hidden layers + Sigmoid output for multi-label.

- Trained separately on both SBERT and TF-IDF representations.

## Training Setup
- Optimizer: Adam

- Loss Function: Binary Cross Entropy (BCELoss)

- Metrics:

    - ROC AUC Score

    - Hamming Loss

    - F-1 Score, Recall and ROC-AUC Curves (Precision was not used due to imbalanced )
- 80-20 Train-Test Split

## Evaluation & Results
### Metrics Comparison
![image](https://github.com/user-attachments/assets/3e031ac9-7dd2-4400-9124-e215e57baf05)
![image](https://github.com/user-attachments/assets/d63ab72e-03bf-4a9c-980d-7992bece64bb)
![image](https://github.com/user-attachments/assets/5a621c49-e876-42c7-8934-0c502fcd432a)

### ROC Curves for all genres
![image](https://github.com/user-attachments/assets/a23cf95f-0c66-44f8-a8ec-dc6e418fbd0a)
![image](https://github.com/user-attachments/assets/88094f1b-e217-41a1-8143-b41a6fc3c538)

### Classification Report
![image](https://github.com/user-attachments/assets/abd258e7-59e9-44f0-a2c7-943fb49fabfc)

![image](https://github.com/user-attachments/assets/9af49157-4b11-4400-9215-0b1f685eb7e3)

## Observations
- SBERT and TFIDF go toe-to-toe in performance despite SBERT having a higher (very small) Hamming Loss.

- Here's a small example of the genre prediction for 5 animes from the test set by both models: 
```
Anime Title: Xian Wu Chuan
Synopsis (Preprocessed): loyal disciple ye chen dedicated guard spiritual medicine field sect fight enemy spiritual field destroyed loyalty dedicating sect could save loyalty thought obtained peer lover could save betrayal thus shamelessly banished sect help flame falling heaven ye chen began develop stronger cultivator battled opponent unfolded legendary life rewrote story
Original Genres: Action, Adventure, Fantasy, Historical
Predicted Genres (SBERT): Action, Adventure, Fantasy
Predicted Genres (TF-IDF): Action, Adventure, Fantasy, Historical
--------------------------------------------------------------------------------
Anime Title: Non Non Biyori: Okinawa e Ikukoto ni Natta
Synopsis (Preprocessed): spending summer day department store suguru koshigaya win lottery grand prizefour ticket okinawa filled awe excitement girl asahigaoka various thing prepare trip practicing ride airplane buying travel essential convenience store everything beforehand enjoy time okinawa fullest extent departure familiar scenery asahigaoka new experience renge miyauchi stop pondering perspective world may change day trip draw near promise made
Original Genres: Everyday Life, School
Predicted Genres (SBERT): Everyday Life
Predicted Genres (TF-IDF): Everyday Life
--------------------------------------------------------------------------------
Anime Title: Yuusha Exkaiser
Synopsis (Preprocessed): alien plan invade earth young boy named kouta must team giant robot outer space exkaiser order save world friend
Original Genres: Action, Science Fiction, Fighting Robots
Predicted Genres (SBERT): Fighting Robots, Science Fiction
Predicted Genres (TF-IDF): Action, Adventure, Fighting Robots, Science Fiction
--------------------------------------------------------------------------------
Anime Title: Xinghe Zhizun
Synopsis (Preprocessed): dazzling blue earth star first door star clan accidentally got world treasure star map attracted eye lord world order get star map lord world induces high level star clan go holy star domain imprisons holy star domain order regain former glory chu xinghe younger suzerainty seek imprisoned father highlevel star clan lead several disciple gate break twelve star circle defeat twelve star spirit lord world eventually rescue father people star clan lord world
Original Genres: Action, Fantasy, Historical
Predicted Genres (SBERT): Action, Adventure, Fantasy
Predicted Genres (TF-IDF): Action, Fantasy
--------------------------------------------------------------------------------
Anime Title: Shiritsu Araiso Koutougakkou Seitokai Shikkoubu
Synopsis (Preprocessed): kubota makoto tokitoh minoru character kazuya minekuras manga wild adaptorthough reference made darker storyline wa lighthearted animeare muscle high school allpowerful student council defend student body disordergenerated human demonswhile avoiding class
Original Genres: Comedy
Predicted Genres (SBERT): Comedy, School
Predicted Genres (TF-IDF): Comedy, School
--------------------------------------------------------------------------------
```
## Requirements
```
pip install iterative-stratification
pip install sentence-transformers
```
## How to Run
   1. Clone the repo or download the notebook.
   2. Place the dataset in the correct path (`/directory/anime_dataset_preprocessed.csv` or update accordingly).
   3. Run all cells in `animegenreclassfn.ipynb`.
   4. Stonks 📈
## Future Improvements
- Try out transformers like DistilBERT or RoBERTa.

- Use attention mechanisms or multi-task heads for better genre modeling.

- Hyperparameter tuning for more optimized training.
## Author
Prakhar Srivastava - watches too much anime
