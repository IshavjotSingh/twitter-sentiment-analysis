# twitter-sentiment-analysis
Twitter Sentiment Analysis using NLP

📌 Project Overview

This project implements Sentiment Analysis on Twitter data using Natural Language Processing (NLP) and Machine Learning techniques.
The objective is to classify tweets into Positive, Neutral, or Negative sentiments.

The project is implemented in Google Colab and follows all requirements mentioned in the task document.

🎯 Objectives

Perform sentiment analysis on tweet text
Clean and preprocess raw Twitter data
Classify sentiments into three categories
Visualize sentiment distribution
Use a simple machine learning model

  Tech Stack

Python
Pandas
NLTK
TextBlob
Scikit-learn
Matplotlib
Google Colab

📂 Repository Structure
.
├── Twitter_Sentiment_Analysis.ipynb
└── README.md

📊 Dataset

The dataset consists of sample Twitter-style text data containing user opinions and feedback.
It is used to simulate real-world Twitter sentiment analysis.

 Project Workflow
1️⃣ Data Loading

Tweets are loaded into a Pandas DataFrame.

2️⃣ Text Preprocessing

Convert text to lowercase
Tokenization using NLTK
Removal of stopwords
Removal of punctuation

3️⃣ Sentiment Labeling

Sentiment polarity is calculated using TextBlob
Tweets are labeled as:

Positive
Neutral
Negative

4️⃣ Feature Extraction

Text data is converted into numerical format using CountVectorizer (Bag of Words)

5️⃣ Model Training

A Naive Bayes (MultinomialNB) classifier is trained
Dataset is split into training and testing sets

6️⃣ Evaluation

Model performance is evaluated using classification metrics:

Precision
Recall
F1-score

7️⃣ Visualization
Sentiment distribution is visualized using a bar chart

📈 Results

Successful classification of tweets into sentiment categories
Clear visualization of sentiment distribution
Effective performance for a small dataset

▶️ How to Run the Project
Option 1: Google Colab (Recommended)

Open Google Colab
Upload the notebook file
Run all cells sequentially

Option 2: Local System
pip install pandas nltk textblob scikit-learn matplotlib
Run the notebook using Jupyter Notebook or VS Code.

🧠 Model Used

Naive Bayes Classifier

Simple and efficient
Suitable for text classification tasks
Commonly used in NLP applications

📌 Key Learnings

Natural Language Processing techniques
Text preprocessing using NLTK
Sentiment analysis using polarity scores
Machine learning-based text classification
Data visualization for insights

📝 Conclusion

This project demonstrates an end-to-end Twitter Sentiment Analysis pipeline, covering text preprocessing, sentiment classification, machine learning modeling, and visualization, strictly following the given task requirements.

👤 Author

Ishavjot Singh
