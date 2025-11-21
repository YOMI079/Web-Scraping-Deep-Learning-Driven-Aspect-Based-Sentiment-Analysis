# Deep Learning Driven Aspect-Based Sentiment Analysis (ABSA)

## Project Overview
This project implements an end-to-end pipeline for **Aspect-Based Sentiment Analysis (ABSA)** on mobile phone reviews from Flipkart. Unlike standard sentiment analysis which gives a single score for a sentence, ABSA identifies specific features (aspects) mentioned in the text (e.g., "Camera", "Battery") and determines the sentiment associated with them.

## Key Features
* **Web Scraping**: Automated extraction of reviews using Python.
* **Advanced Text Cleaning**: Handling Unicode normalization, emoji decoding, and spelling correction.
* **Unsupervised Topic Modeling**: Using **Latent Dirichlet Allocation (LDA)** to automatically discover product aspects from unlabeled text.
* **Data Augmentation**: Utilizing `nlpaug` to balance the dataset by generating synthetic negative reviews via antonym replacement.
* **Deep Learning Model**: A PyTorch-based **Bi-Directional LSTM** with **FastText** embeddings for sentiment classification.

## Technical Details
For a deep dive into the methodology, algorithms (LDA, LSTM), and preprocessing steps (Unicode normalization, lemmatization), please read the [Technical Explanation](TECHNICAL_EXPLANATION.md).

## Dataset
The data is scraped from Flipkart cell phone reviews. It includes product specifications and user reviews (text, title, rating).
* **Size**: ~50,000 reviews (before cleaning/augmentation).
* **Identified Aspects**: Phone, Camera, Battery, Delivery, Processor.

## Installation & Usage

### Prerequisites
* Python 3.7+
* PyTorch
* Gensim
* NLTK
* Spacy
* NLPAug

### Running the Project
1.  **Clone the repository**
2.  **Install dependencies**:
    ```bash
    pip install torch pandas numpy matplotlib seaborn nltk spacy gensim nlpaug demoji langdetect
    python -m spacy download en_core_web_sm
    ```
3.  **Run the Notebook**:
    Open `ABSA_Code_Results.ipynb` in Jupyter Notebook or Google Colab.

### Inference Example
The model takes a review string and outputs the primary aspect and the sentiment.

```python
sample = "I am really impressed with the phone's great battery backup."
# Output: "The reviewer is talking Positively about the battery of the phone."
