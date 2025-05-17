# Naive-Bayes
This project uses the Naive Bayes algorithm to classify SMS messages as spam or ham (not spam). It processes raw text data using NLP techniques, converts messages into feature vectors, and trains a classifier to detect spam. It’s fast, accurate, and ideal for real-time spam filtering.
# ✉️ SMS Spam Detection using Naive Bayes

This project applies the **Naive Bayes algorithm** to detect spam messages from a dataset of SMS texts. It's a classic NLP classification task where probabilistic reasoning helps filter out unwanted messages with high accuracy and efficiency.

---

## 📌 Project Highlights

- Preprocesses raw text using **tokenization**, **lowercasing**, and **stopword removal**
- Converts text to numeric format using **TF-IDF Vectorization**
- Uses **Multinomial Naive Bayes** for training
- Evaluates with **accuracy, confusion matrix, and classification report**

---

## 📊 Dataset

- **Name**: SMS Spam Collection
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Size**: 5,572 labeled messages
- **Labels**: `spam`, `ham` (not spam)

---

## 📈 Algorithm – Naive Bayes

Naive Bayes is a probabilistic classifier based on Bayes’ Theorem. It assumes feature independence and is particularly effective for text classification tasks. It calculates the probability of a message being spam given the words it contains.

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/naive-bayes-spam-filter.git
   cd naive-bayes-spam-filter

2.	Install dependencies:
   ```bash
   pip install pandas sklearn matplotlib seaborn nltk
   ```
3.	Download the dataset from the UCI Repository and place it in the project directory as spam.csv.
   
5.	Run the classifier:
   ```bash
   python spam_filter.py
   ```
📊 Output
	•	Confusion Matrix
	•	Accuracy Score
	•	Classification Report
	•	Bar chart of prediction results

⸻

🧠 Why Naive Bayes?
	•	Fast and scalable for large text data
	•	Handles high-dimensional inputs like word vectors
	•	Assumes conditional independence — a simplification that works surprisingly well in practice

⸻

👨‍💻 Author
	•	Code and logic written by Siddharth Vishwanath
	•	Refined, optimized, and structured with the help of AI Assistant

