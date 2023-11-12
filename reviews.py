import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC  
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Words to remove from stopwords list
words_to_remove = ["can", "nor", "hadn", "couldn", "wasn't", "don't", "any", "doesn", "didn't", "couldn't", "down",
                   "isn't", "shouldn't", "above", "did", "during", "again", "won't", "against", "won", "mustn't",
                   "needn't", "didn't", "couldn't", "down", "isn't", "shouldn't", "above", "did", "during", "again",
                   "won't", "against", "won", "not", "no"]

# Remove specified words from the stopwords list
for word in words_to_remove:
    if word in stop_words:
        stop_words.remove(word)

# Labels for reviews

labels_list = ["TRUTHFULPOSITIVE", "TRUTHFULNEGATIVE", "DECEPTIVENEGATIVE", "DECEPTIVEPOSITIVE"]

# Function to preprocess reviews

def preprocess(review: str):

    review = review.lower()

    words = nltk.word_tokenize(review)

    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]   

    preprocessed_review = ' '.join(words)

    return preprocessed_review

# Lists to store the extracted labels and reviews
extracted_labels = []
extracted_reviews = []

with open('train.txt', 'r') as file:
    # Read the content of the file
    content = file.read()
    
    # Split the content by two or more newline characters to get paragraphs
    paragraphs = [p for p in content.split('\n') if p]

    for paragraph in paragraphs:
        label_found = None
        review_found = None

        for l in labels_list:
            if paragraph.startswith(l):
                label_found = l
                review_found = paragraph[len(label_found):].strip()
                break

        if label_found and review_found:
            extracted_labels.append(label_found)
            extracted_reviews.append(preprocess(review_found))
df = pd.DataFrame({'Labels': extracted_labels,'Reviews': extracted_reviews})

file.close()

test = []

with open('test_just_reviews.txt', 'r') as file:
    # Read the content of the file
    content = file.read()
    paragraphs = content.split('\n')
    paragraphs.pop(-1) # Last element is empty
    for paragraph in paragraphs :
        test.append(preprocess(paragraph))

original_stderr = sys.stderr

#1. Preprocess the Training Data

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(df['Reviews'])
y_train = df['Labels']

# -----------------------------------------------------------------------------
# -------------------- Code used to get results for paper ---------------------
# -----------------------------------------------------------------------------

# X = vectorizer.fit_transform(df['Reviews'])
# y = df['Labels']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 17)

#2. Train a Model (Best one)

clf = SVC(kernel = 'linear') # accuracy 0.8785714285714286

# -----------------------------------------------------------------------------
# -------------------- Other models used for this project ---------------------
# -----------------------------------------------------------------------------

# clf = LogisticRegression(max_iter=1000) # accuracy 0.8535714285714285
# clf = DecisionTreeClassifier(max_depth=3, random_state=42) # accuracy 0.4857142857142857
# clf = KNeighborsClassifier(n_neighbors = 7) # accuracy 0.6428571428571429
# clf = MultinomialNB() # accuracy 0.7928571428571428
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42) # accuracy 0.6892857142857143

clf.fit(X_train, y_train)

#3. Preprocess the Test Data

X_test = vectorizer.transform(test)

#4. Test the Model

y_pred = clf.predict(X_test)

#5. Print the predictions for the test data on result.txt
with open('results.txt', 'w') as f:
    sys.stderr = f
    for i in range(len(y_pred)):
        if i != len(y_pred)-1:
            print(y_pred[i], file=sys.stderr)
        else:
            print(y_pred[i], file=sys.stderr, end='')

# -----------------------------------------------------------------------------
# -------------------- Code used to get results for paper ---------------------
# -----------------------------------------------------------------------------

# #6. Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy", accuracy)

# #7. Accuracy by label
# accuracies = []
# for label in labels_list:
#     true_indices = [i for i, true_label in enumerate(y_test) if true_label == label]
#     pred_indices = [i for i, pred_label in enumerate(y_pred) if pred_label == label]
    
#     true_positive_count = len(set(true_indices) & set(pred_indices))
#     total_true_count = len(true_indices)
    
#     accuracy = (true_positive_count / total_true_count)*100 if total_true_count > 0 else 0
#     accuracies.append(accuracy)

# plt.figure(figsize=(8, 6))
# bars = plt.bar(labels_list, accuracies, color='skyblue')

# for bar, accuracy in zip(bars, accuracies):
#     plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.02, f'{accuracy:.2f}', ha='center')

# plt.xlabel('Labels')
# plt.ylabel('Accuracy')
# plt.title('Accuracy per Label')
# plt.ylim(0, 100)  # Set y-axis limit between 0 and 1 for accuracy range
# plt.savefig('images/accuracy_per_label.png')
# plt.clf()

# #8. Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)

# plt.figure(figsize=(10, 10))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_list, yticklabels=labels_list, annot_kws={"size": 16})
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.savefig('images/confusion_matrix_SVC.png')

# -----------------------------------------------------------------------------
# -------------------- Code used to find wrong ouputs -------------------------
# -----------------------------------------------------------------------------

# inputs = vectorizer.inverse_transform(X_test)

# inputs = [' '.join(words) for words in inputs]

# def are_words_included(str1, str2):
#     # Split the strings into words and convert them to sets
#     set1 = set(str1.split())
#     set2 = set(str2.split())

#     # Check if all words in set1 are included in set2
#     return set1.issubset(set2)

# for input, prediction, label in zip(inputs, y_pred, y_test):
#     if prediction != label: 
#         for review in df['Reviews']:
#             if are_words_included(input, review):
#                 print(review, 'has been classified as ', prediction, 'and should be ', label, end='\n')