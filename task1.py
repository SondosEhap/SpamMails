# reading data
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import statistics
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns 
# download nltk
import nltk
nltk.download('all')
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import GaussianNB

#=============================================load data=======================================
data = pd.read_csv(r'C:\Users\Alrahma\Downloads\data_spam.csv', encoding='latin-1')
print(data.head())

# drop unnecessary columns and rename cols
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
data.columns = ['label', 'text']
# ==========================================visualization using pie&bar charts ========================
#pie chart
print(data.groupby('label').size().plot(kind='pie',autopct='%.0f', shadow = True,colors=['darkblue','darkcyan']))

#bar plot
print(data['label'].value_counts(normalize = True).plot(kind='bar',xlabel='Label',ylabel='count',color=['sienna','brown']))
# ==============================================text preprocessing==================================
# check missing values
print (data.isna().sum())

# check data shape
print(data.shape)
# create a list text
text = list(data['text'])

# preprocessing loop
lemmatizer = WordNetLemmatizer()
stemmer= PorterStemmer()
corpus = []

for i in range(len(text)):
    r = re.sub('[^a-zA-Z]', ' ', text[i])#replace any not  charactar to space
    r = r.lower()
    
#Tokanization
    r = r.split()#convert string to list
    r = [word for word in r if word not in stopwords.words('english')]#add the word only if irt not a stop word

#lemmatizatoin
    r = [lemmatizer.lemmatize(word) for word in r]#convert the word to its origin

#steeming
    r=[ stemmer.stem(word)for word in r]#steeming 
    r = ' '.join(r)#join all words in string 

#append data to the array    
    corpus.append(r)

#assign corpus to data['text']
data['text'] = corpus
print(data.head())

#=========================================splitting data int training and test===============
# Create Feature and Label sets
X = data['text']
y = data['label']

# train test split (66% train - 33% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

print('Training Data :', X_train.shape)
print('Testing Data : ', X_test.shape)

# Train Bag of Words model
cv = CountVectorizer()#convert text to numeric data
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)# transform X_test using CV
print(X_train_cv)
#======================================perform statistics on text column====================
T=cv.transform(X)
T.toarray()
mean_value=np.mean(T)
mediann_value=np.median(T.toarray())
#mod_value=statistics.mode(T)
print('mean_value',mean_value)
print('mediann_value',mediann_value)
#print('mod_value',mod_value)

#============================== Logistic Regression model==================================

lr = LogisticRegression()#model
lr.fit(X_train_cv, y_train)

# generate predictions
predictions = lr.predict(X_test_cv)
print (predictions)

# confusion matrix
df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])
print("---------------logistic regression accuracy---------------------------")

print(df)
print(classification_report(y_test,predictions)) 
print(accuracy_score(y_test,predictions)*100)

#=======================================knn algorithm=========================================
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_cv, y_train)

prediction = knn.predict(X_test_cv)

df = pd.DataFrame(metrics.confusion_matrix(y_test,prediction), index=['ham','spam'], columns=['ham','spam'])
accuracy = accuracy_score(y_test,prediction)
print("-------------------------KNN accuracy-------------------------------")

print(df)
print(classification_report(y_test,prediction)) 
print("Accuracy:", accuracy*100)
#plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
#plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
#plt.show()
#=================================naive base==========================================

# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(X_train_cv.toarray(), y_train)

# Predict Output
predicted = model.predict(X_test_cv.toarray())

df = pd.DataFrame(metrics.confusion_matrix(y_test,predicted), index=['ham','spam'], columns=['ham','spam'])
accuracy = accuracy_score(y_test,predicted)
print("-------------------------Naive bayes accuracy-------------------------")

print(df)
print(classification_report(y_test,predicted)) 
print("Accuracy:", accuracy*100)
