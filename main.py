#%%
# ==============================
# We import our libary
# ==============================
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import en_core_web_lg
import spacy
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier,RidgeClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
#%%
df = pd.read_csv('C:/Users/Msi/Desktop/CALISMALAR/NLP/data/emotion-labels-train.csv')
#%%
df.head(5)
#%%
df.info()
df=df.dropna()
#%%

sns.countplot(x='label', data=df, palette='viridis')
plt.xticks(rotation=40)
plt.show()
#%%
df['label'].unique()
df["label"] = df["label"].replace({'joy':0,'fear':1,'anger':2,'sadness':3})
#%%
nlp = spacy.load("en_core_web_lg")
stop_words = nlp.Defaults.stop_words
punctuations = string.punctuation
#%%
def spacy_tokenizer(sentence):
    if type(sentence) == float:
        sentence=sentence
    else:
        doc = nlp(sentence)
        mytokens = [ word.lemma_.lower().strip() for word in doc ]
        mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
        sentence = " ".join(mytokens)
    return sentence
#%%
df['tokenized_text'] = df['text'].apply(spacy_tokenizer)
print(df)
#%%
vec = CountVectorizer()
df_vec = vec.fit_transform(df["tokenized_text"])
 #%%
clf1 = SGDClassifier()
clf2 = XGBClassifier()
clf3 = LogisticRegression()
clf4=  RidgeClassifier()
clf5= SVC()
eclf = VotingClassifier(estimators=[('SGD', clf1), ('XGB', clf2),('LG', clf3),
                                    ('RC', clf4),('SVC', clf5)],voting='hard')

for clf, label in zip([clf1,clf2,clf3,clf4,clf5,eclf],
                      ['SGD','XGB','LG','RC', 'SVC','Ensemble']):
    scores = cross_val_score(clf, df_vec, df["label"], scoring='accuracy', cv=8)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
#%%
Voting_model=eclf.fit(df_vec, df["label"])
print(Voting_model)
#%%
df_test = pd.read_csv('C:/Users/Msi/Desktop/CALISMALAR/NLP/data/emotion-labels-test.csv')

#%%
df_test['tokenized_text'] = df_test['text'].apply(spacy_tokenizer)
df_test
df_test_vec = vec.transform(df_test["tokenized_text"])
#%%
Voting_pred = Voting_model.predict(df_test_vec)
Voting_pred = pd.DataFrame(Voting_pred)
#%%
y_test_df=pd.DataFrame(df_test['label'])
print(y_test_df)
#%%
my_text = ("She clenched her fists, jaw tight, eyes narrowed, "
           "expressing her frustration "
           "without uttering a word")

my_text_vec = vec.transform([spacy_tokenizer(my_text)])

prediction = Voting_model.predict(my_text_vec)

if prediction == 0:
    print("Prediction: Joy")
elif prediction == 1:
    print("Prediction: Fear")
elif prediction == 2:
    print("Prediction: Anger")
elif prediction == 3:
    print("Prediction: Sadness")
#%%








