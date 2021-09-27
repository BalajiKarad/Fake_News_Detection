#!/usr/bin/env python
# coding: utf-8

# In[1]:


from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import numpy as np # linear algebra
import pandas as pd #data processing

import os
import re
import nltk


# In[2]:


train=pd.read_csv('./fake-news/train.csv')
test=pd.read_csv('./fake-news/test.csv')


# In[3]:


print(train.shape, test.shape)


# In[4]:


print(train.isnull().sum())
print('************')
print(test.isnull().sum())


# In[5]:


test=test.fillna(' ')
train=train.fillna(' ')
test['total']=test['title']+' '+test['author']+test['text']
train['total']=train['title']+' '+train['author']+train['text']


# # Creating Wordcloud Visuals

# In[6]:


real_words = ''
fake_words = ''
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in train[train['label']==1].total: 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    real_words += " ".join(tokens)+" "

for val in train[train['label']==0].total: 
      
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    fake_words += " ".join(tokens)+" "


# In[7]:


wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(real_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# # Cleaning and preprocessing 

# # 1. Regex

# In[8]:


#Remove punctuations from the String  
s = "!</> hello please$$ </>^s!!!u%%bs&&%$cri@@@be^^^&&!& </>*to@# the&&\ cha@@@n##%^^&nel!@# %%$"


# In[9]:


s = re.sub(r'[^\w\s]','',s)


# In[10]:


print(s)


# # 2. Tokenization

# In[11]:


#Downloading nltk data
nltk.download('punkt')


# In[12]:


nltk.word_tokenize("Hello how are you")


# # 3. StopWords

# In[13]:


from nltk.corpus import stopwords

stop_words = stopwords.words('english')
print(stop_words)


# In[14]:


sentence = "Covid-19 pandemic has impacted many countries and what it did to economy is very stressful"


# In[15]:


words = nltk.word_tokenize(sentence)
words = [w for w in words if w not in stop_words]


# In[16]:


words


# # 4. Lemmatization

# In[17]:


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

input_str="been had done languages cities mice"


# In[18]:


#Tokenize the sentence
input_str=nltk.word_tokenize(input_str)

#Lemmatize each word
for word in input_str:
    print(lemmatizer.lemmatize(word))


# # Let's Apply

# In[19]:


lemmatizer=WordNetLemmatizer()
for index,row in train.iterrows():
    filter_sentence = ''
    
    sentence = row['total']
    sentence = re.sub(r'[^\w\s]','',sentence) #cleaning
    
    words = nltk.word_tokenize(sentence) #tokenization
    
    words = [w for w in words if not w in stop_words]  #stopwords removal
    
    for word in words:
        filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()
        
    train.loc[index,'total'] = filter_sentence


# In[20]:


train = train[['total','label']]


# # Applying NLP Techniques

# In[21]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[22]:


X_train = train['total']
Y_train = train['label']


# # Bag-of-words / CountVectorizer

# In[23]:


corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())


# In[24]:


print(X.toarray())


# # TF-iDF Vectorizer

# In[25]:


def vectorize_text(features, max_features):
    vectorizer = TfidfVectorizer( stop_words='english',
                            decode_error='strict',
                            analyzer='word',
                            ngram_range=(1, 2),
                            max_features=max_features
                            #max_df=0.5 # Verwendet im ML-Kurs unter Preprocessing                   
                            )
    feature_vec = vectorizer.fit_transform(features)
    return feature_vec.toarray()


# In[26]:


tfidf_features = vectorize_text(['hello how are you doing','hi i am doing fine'],30)


# In[27]:


tfidf_features


# # Let's Apply

# In[28]:


#Feature extraction using count vectorization and tfidf.
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(X_train)
freq_term_matrix = count_vectorizer.transform(X_train)
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)


# In[29]:


tf_idf_matrix


# # Modelling

# In[30]:


test_counts = count_vectorizer.transform(test['total'].values)
test_tfidf = tfidf.transform(test_counts)

#split in samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, Y_train, random_state=0)


# # Logistic Regression

# In[32]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)
pred = logreg.predict(X_test)
print('Accuracy of Lasso classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Lasso classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))
from sklearn.naive_bayes import MultinomialNB
cm = confusion_matrix(y_test, pred)
cm


# # MultinomialNB

# In[33]:


from sklearn.naive_bayes import MultinomialNB

NB = MultinomialNB()
NB.fit(X_train, y_train)
pred = NB.predict(X_test)
print('Accuracy of NB  classifier on training set: {:.2f}'
     .format(NB.score(X_train, y_train)))
print('Accuracy of NB classifier on test set: {:.2f}'
     .format(NB.score(X_test, y_test)))
cm = confusion_matrix(y_test, pred)
cm


# # Pipeline

# In[34]:


#Assiging the variables again as once transformed vectors can't be transformed again using pipeline.
X_train = train['total']
Y_train = train['label']


# In[35]:


from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[36]:


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf', linear_model.LogisticRegression(C=1e5)),
])


# In[37]:


pipeline.fit(X_train, Y_train)


# In[38]:


pipeline.predict(["flynn hillary clinton big woman campus breitbart daniel j flynnever get feeling life circle roundabout rather head straight line toward intended destination hillary clinton remains big woman campus leafy liberal wellesley massachusetts everywhere else vote likely inauguration dress remainder day way miss havisham forever wore wedding dress speaking great expectations hillary rodham overflowed 48 year ago first addressed wellesley graduating class the president college informed gathered 1969 student needed debate far i could ascertain spokesman kind like democratic primary 2016 minus term unknown even seven sisters school i glad miss adams made clear i speaking today u 400 u miss rodham told classmate after appointing edger bergen charlie mccarthys mortimer snerds attendance bespectacled granny glass awarding matronly wisdom least john lennon wisdom took issue previous speaker despite becoming first win election seat u s senate since reconstruction edward brooke came criticism calling empathy goal protestors criticized tactic though clinton senior thesis saul alinsky lamented black power demagogue elitist arrogance repressive intolerance within new left similar word coming republican necessitated brief rebuttal trust rodham ironically observed 1969 one word i asked class rehearsal wanted say everyone came said talk trust talk lack trust u way feel others talk trust bust what say what say feeling permeates generation perhaps even understood distrusted the trust bust certainly busted clintons 2016 plan she certainly even understand people distrusted after whitewater travelgate vast conspiracy benghazi missing email clinton found distrusted voice friday there load compromising road broadening political horizon and distrust american people trump edged 48 percent 38 percent question immediately prior novembers election stood major reason closing horizon clinton described vanquisher supporter embracing lie con alternative fact assault truth reason she failed explain american people chose lie truth as history major among today know well people power invent fact attack question mark beginning end free society offered that hyperbole like many people emerge 1960s hillary clinton embarked upon long strange trip from high school goldwater girl wellesley college republican president democratic politician clinton drank time place gave degree more significantly went idealist cynic comparison two wellesley commencement address show way back lamented long leader viewed politics art possible challenge practice politics art making appears impossible possible now big woman campus odd woman white house wonder current station even possible why arent i 50 point ahead asked september in may asks isnt president the woman famously dubbed congenital liar bill safire concludes lie mind getting stood election day like finding jilted bride wedding day inspires dangerous delusion"])


# In[39]:


#saving the pipeline
filename = 'pipeline.sav'
joblib.dump(pipeline, filename)


# In[40]:


filename = './pipeline.sav'


# # Prediction

# In[45]:


loaded_model = joblib.load(filename)
result = loaded_model.predict(["Written by Shaun Bradley   Mandatory vaccinations are about to open up a new frontier for government control. Through the war on drugs, bureaucrats arbitrarily dictate what people can and canâ€™t put into their bodies, but that violation pales in comparison to forcibly medicating millions against their will. Voluntary and informed consent are essential in securing individual rights, and without it, self-ownership will never be respected. The liberal stronghold of California is trailblazing the encroaching new practice and recently passed laws mandating that children and adults must have certain immunizations before being able to attend schools or work in certain professions. The longstanding religious and philosophical exemptions that protect freedom of choice have been systematically crushed by the state. Californiaâ€™s Senate Bill 277 went into effect on July 1st, 2016, and marked the most rigid requirements ever instituted for vaccinations. The law forces students to endure a total of 40 doses to complete the 10 federally recommended vaccines while allowing more to be added at any time. Any family that doesnâ€™t go along will have their child barred from attending licensed day care facilities, in-home daycares, public or private schools, and even after school programs. Over the years, California has developed a reputation for pushing vaccines on their youth. Assembly Bill 499 was passed in 2011 and lowered the age of consent for STD prevention vaccines to just 12 years old. Included in the assortment of shots being administered was the infamous Gardasil , which just a few years later was at the center of a lawsuit that yielded the victims a $6 million settlement from the US government, which paid out funds from the National Vaccine Injury Compensation Program . The Vaccinate All Children Act of 2015 is an attempt to implement this new standard nationwide, and although it has stalled in the House, it will likely be reintroduced the next time the country is gripped by the fear of a pandemic. The debate surrounding vaccinations is commonly framed as a moral struggle between the benefits to the collective and the selfish preferences of the individual. But since the outbreak scares of Zika , measles , and ebola , the rhetoric has taken a turn toward authoritarianism. Itâ€™s commonly stated by the CDC and most mainstream doctors that the unvaccinated are putting the health of everyone else at risk, but the truth isnâ€™t so black and white . The herd immunity theory has been consistently used to validate the expansion of vaccine programs, but it still doesnâ€™t justify the removal of choice from the individual. The classic exchange of freedom for perceived safety is a no brainer for the millions of Americans who are willing to use government to strap their neighbors down and forcibly inject them for the greater good. Anyone who expresses concern about possible side effects is immediately branded as conspiratorial or anti-science. Yet controversial claims that "])
print(result) 


# In[ ]:




