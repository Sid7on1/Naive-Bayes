import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import zipfile
import urllib.request

# Step 1: Download and extract dataset
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
zip_path = "smsspamcollection.zip"
dataset_path = "SMSSpamCollection"

if not os.path.exists(dataset_path):
    print("📦 Downloading dataset...")
    urllib.request.urlretrieve(dataset_url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    os.remove(zip_path)

# Step 2: Parse safely with validation
data = []
with open(dataset_path, 'r', encoding='ISO-8859-1') as file:
    for line in file:
        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
            data.append(parts)

df = pd.DataFrame(data, columns=['label', 'message'])

# Step 3: Clean and validate
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.dropna(inplace=True)  # Drop any rows with NaN just in case

assert df['label'].isnull().sum() == 0, "❌ Label column still has NaN!"
assert df['message'].isnull().sum() == 0, "❌ Message column still has NaN!"

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42)

# Step 5: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test_vec)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Naive Bayes SMS Spam Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()