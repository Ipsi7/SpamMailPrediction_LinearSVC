import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


#Data PreProcessing
mail_data = pd.read_csv('SMSSpamCollection.txt', sep ='\t', header=None, names = ['Category', 'Message'])
#print(mail_data.head())
#print(mail_data.shape)

mail_data.loc[mail_data['Category']=='spam', 'Category',]=0
mail_data.loc[mail_data['Category']=='ham', 'Category',]=1

X = mail_data['Message']
Y = mail_data['Category']

#print(X)
#print(Y)


#train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=3)

#Feature feature_extraction
#transforming text data to vector
#convert all text to lower case

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase = 'True')
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#Training the model  --- SVM


model=LinearSVC()
model.fit(X_train_features, Y_train)

#Evaluation of the model

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('accuracy_on_training_data', accuracy_on_training_data)

#now check accuracy on test data which our model have not seen

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('accuracy_on_testing_data', accuracy_on_test_data)

#Prediction on new mail

input_mail = ["I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."]

input_mail_features = feature_extraction.transform(input_mail)


prediction = model.predict(input_mail_features)
#print(prediction)

if (prediction[0]==1):
    print('It is a HAM MAIL')
else:
    print('It is a Spam Mail')
