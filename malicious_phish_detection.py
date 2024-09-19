#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import itertools
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from lightgbm import LGBMClassifier
import os
import seaborn as sns
from wordcloud import WordCloud
import joblib
import pickle


# In[2]:


df = pd.read_csv('dataset/malicious_phish.csv')

print(df.shape)
df


# In[3]:


df.type.value_counts()


# In[4]:


df_phish = df[df.type=='phishing']
df_malware = df[df.type=='malware']
df_deface = df[df.type=='defacement']
df_benign = df[df.type=='benign']


# In[5]:


phish_url = " ".join(i for i in df_phish.url)
wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(phish_url)
plt.figure(figsize=(12,14),facecolor='k')
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[6]:


malware_url = " ".join(i for i in df_malware.url)
wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(malware_url)
plt.figure(figsize=(12,14),facecolor='k')
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[7]:


deface_url = " ".join(i for i in df_deface.url)
wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(deface_url)
plt.figure(figsize=(12,14),facecolor='k')
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[8]:


benign_url = " ".join(i for i in df_benign.url)
wordcloud = WordCloud(width=1600, height=800, colormap='Paired').generate(benign_url)
plt.figure(figsize=(12,14),facecolor='k')
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# FEATURE ENGINEERING

# In[9]:


import re
#use of IP or not in domain 
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  #IPV4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'  #IPV4 hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)#IPV6
    # match group or not
    if match:
        #print match.group()
        return 1
    else:
        #print 'No matching pattern found'
        return 0

df['use_of_ip']= df['url'].apply(lambda i: having_ip_address(i))


# In[10]:


from urllib.parse import urlparse

def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        #print match.group()
        return 1
    else:
        #print 'No matching pattern found'
        return 0
    
df['abnormal_url'] = df['url'].apply(lambda i: abnormal_url(i))


# In[11]:


from googlesearch import search


# In[12]:


def google_index(url):
    site = search(url, 5)
    return 1 if site else 0	
df['google_index'] = df['url'].apply(lambda i: google_index(i))


# In[13]:


def count_dot(url):
    count_dot = url.count('.')
    return count_dot

df['count.'] = df['url'].apply(lambda i: count_dot(i))
#df.head()
df


# In[14]:


def count_www(url):
    url.count('www')
    return url.count('www')

df['count-www'] = df['url'].apply(lambda i: count_www(i))

def count_atrate(url):
    return url.count('@')

df['count@'] = df['url'].apply(lambda i: count_atrate(i))

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

df['count_dir'] = df['url'].apply(lambda i: no_of_dir(i))

def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count("//")

df['count_embed_domian'] = df['url'].apply(lambda i: no_of_embed(i))

def shortening_service(url):
    match = re.search ('bit/.ly|goo/.gl|shorte/.st|go2l/.ink|x/.co|ow/.ly|t/.co|tinyurl|tr/.im|is/.gd|cli/.gs|'
                      'yfrog/.com|migre/.me|ff/.im|tiny/.cc|url4/.eu|twit/.ac|su/.pr|twurl/.nl|snipurl/.com|'
                      'short/.to|BudURL/.com|ping/.fm|post/.ly|Just/.as|bkite/.com|snipr/.com|fic/.kr|loopt/.us|'
                      'doiop/.com|short/.ie|kl/.am|wp/.me|rubyurl/.com|om/.ly|to/.ly|bit/.do|t/.co|lnkd/.in|'
                      'db/.tt|qr/.ae|adf/.ly|goo/.gl|bitly/.com|cur/.lv|tinyurl/.com|ow/.ly|bit/.ly|ity/.im|'
                      'q/.gs|is/.gd|po/.st|bc/.vc|twitthis/.com|u/.to|j/.mp|buzurl/.com|cutt/.us|u/.bb|yourls/.org|'
                      'x/.co|prettylinkpro/.com|scrnch/.me|filoops/.info|vzturl/.com|qr/.net|1url/.com|tweez/.me|v/.gd|'
                      'tr/.im|link/.zip/.net',
                      url)
    
    if match:
        return 1
    else:
        return 0
    
df['short_url'] = df['url'].apply(lambda i: shortening_service(i))


# In[15]:


def count_https(url):
    return url.count('https')	

df['count-https'] = df['url'].apply(lambda i : count_https(i))

def count_http(url):
    return url.count('http')

df['count-http'] = df['url'].apply(lambda i : count_http(i))


# In[16]:


def count_per(url):
    return url.count('%')

df['count%'] = df['url'].apply(lambda i : count_per(i))

def count_ques(url):
    return url.count('?')

df['count?'] = df['url'].apply(lambda i : count_ques(i))

def count_hyphen(url):
    return url.count('-')
	
df['count-'] = df['url'].apply(lambda i : count_hyphen(i))

def count_equal(url):
    return url.count('=')

df['count='] = df['url'].apply(lambda i : count_equal(i))

def url_length(url):
    return len(str(url))

#length of url
df['url_length'] = df['url'].apply(lambda i: url_length(i))

#Hostname lenght
def hostname_length(url):
    return len(urlparse(url).netloc)

df['hostname_length'] = df['url'].apply(lambda i: hostname_length(i))

df.head()

def suspicious_words(url):
    match = re.search('paypal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                     url)
    if match:
        return 1
    else:
        return 0
df['sus_url'] = df['url'].apply(lambda i: suspicious_words(i))

def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits
    
df['count-digits'] = df['url'].apply(lambda i: digit_count(i))
    
def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters

df['count-letters'] = df['url'].apply(lambda i: letter_count(i))
df


# In[17]:


#importing dependencies  
from urllib.parse import urlparse
from tld import get_tld
import os.path

#First Directory Length
def fd_length(url):
    urlpath = urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0
    
df['fd_length'] = df['url'].apply(lambda i: fd_length(i))


#Length of top level domain
df['tld'] = df['url'].apply(lambda i: get_tld(i,fail_silently=True))
	

def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1
    
df['tld_length'] = df['tld'].apply(lambda i: tld_length(i))


# In[18]:


df


# In[19]:


df.columns


# In[20]:


df['type'].value_counts()


# EDA

# 1.DISTRIBUTION OF USE_OF_IP

# In[21]:


import seaborn as sns
#distribution of use of ip
sns.set(style="darkgrid")
ax = sns.countplot(y="type", data=df,hue="use_of_ip")


# 2.DISTRIBUTION OF ABNORMAL URL

# In[22]:


sns.set(style="darkgrid")	
#ax = sns.catplot(x="type", y="count.",kind="box",data=df, hue="abnormal_url")
ax = sns.countplot(y="type",data=df,hue="abnormal_url")


# 3.DISTRIBUTION OF GOOGLE INDEX

# In[23]:


sns.set(style="darkgrid")	
ax = sns.countplot(y="type",data=df,hue="google_index")


# 4.DISTRIBUTION OF SHORT URL

# In[24]:


sns.set(style="darkgrid")	
ax = sns.countplot(y="type",data=df,hue="short_url")


# 5.DISTRIBUTION OF SUSPICIOUS URL

# In[25]:


sns.set(style="darkgrid")	
ax = sns.countplot(y="type",data=df,hue="sus_url")


# 6.DISTRIBUTION OF COUNT OF [.]DOT

# In[26]:


sns.set(style="darkgrid")	
ax = sns.catplot(x="type",y="count.",kind="box",data=df)


# 7.DISTRIBUTION OF COUNT-WWW

# In[27]:


sns.set(style="darkgrid")	
ax = sns.catplot(x="type",y="count-www",kind="box",data=df)


# 8.DISTRIBUTION OF COUNT@

# In[28]:


sns.set(style="darkgrid")	
ax = sns.catplot(x="type",y="count@",kind="box",data=df)


# 9.DISTRIBUTION OF COUNT_DIR

# In[29]:


sns.set(style="darkgrid")	
ax = sns.catplot(x="type",y="count_dir",kind="box",data=df)


# 10.DISTRIBUTION OF COUNT-HTTPS

# In[30]:


sns.set(style="darkgrid")	
ax = sns.catplot(x="type",y="count-https",kind="box",data=df)


# 11.DISTRIBUTION OF COUNT-HTTP

# In[31]:


sns.set(style="darkgrid")	
ax = sns.catplot(x="type",y="count-http",kind="box",data=df)


# 12.DISTRIBUTION OF HOSTNAME LENGHT

# In[32]:


sns.set(style="darkgrid")	
ax = sns.catplot(x="type",y="hostname_length",kind="box",data=df)


# 13.DISTRIBUTION OF FIRST DIRECTORY LENGTH

# In[33]:


sns.set(style="darkgrid")	
ax = sns.catplot(x="type",y="fd_length",kind="box",data=df)


# 14.DISTRIBUTION OF TOP-LEVEL DOMAIN LENGTH

# In[34]:


sns.set(style="darkgrid")	
ax = sns.catplot(x="type",y="tld_length",kind="box",data=df)


# TARGET ENCODING

# In[35]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
df["type_code"] = lb_make.fit_transform(df["type"])
df["type_code"].value_counts()


# CREATION OF FEATURE & TARGET

# In[36]:


#predicted variables
#filtering out google_index as it has only 1 value
X = df[['use_of_ip','abnormal_url', 'count.', 'count-www','count@',
        'count_dir','count_embed_domian','short_url','count-https',
        'count-http','count%','count?','count-','count=','url_length',
        'hostname_length','sus_url', 'fd_length', 'tld_length','count-digits',
       'count-letters']]

#target variable
y = df['type_code']


# In[37]:


X


# In[38]:


X.columns


# TRAIN TEST SPLIT

# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.2, shuffle=True, random_state=5)


# MODEL BUILDING

# 1.RANDOM FOREST CLASSIFIER - BASE MODEL

# In[40]:


from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,max_features='sqrt')
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test,y_pred_rf,target_names=['benign', 'defacement','phishing','malware']))

score = metrics.accuracy_score(y_test, y_pred_rf)
print("accuracy: %0.3f" % score)


# In[41]:


import joblib
joblib.dump(rf,'model.pkl')


# In[42]:


cm = confusion_matrix(y_test, y_pred_rf)
cm_df = pd.DataFrame(cm,
                    index = ['benign', 'defacement','phishing','malware'], 
                    columns = ['benign', 'defacement','phishing','malware'])
plt.figure(figsize=(8,6))	
sns.heatmap(cm_df, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


# In[43]:


feat_importance = pd.Series(rf.feature_importances_, index=X_train.columns)
feat_importance.sort_values().plot(kind="barh",figsize=(10,6))


# 2.LIGHT GBM CLASSIFIER

# In[44]:


import lightgbm as lgb
lgb = LGBMClassifier(objective='multiclass',boosting_type='gbdt',n_jobs = 5, random_state=5)
LGB_C = lgb.fit(X_train, y_train)

y_pred_lgb = LGB_C.predict(X_test)
print(classification_report(y_test,y_pred_lgb,target_names=['benign', 'defacement','phishing','malware']))

score = metrics.accuracy_score(y_test, y_pred_lgb)
print("accuracy: %0.3f" % score)


# 3.XGBOOST CLASSIFIER

# In[52]:


xgb_c = xgb.XGBClassifier(n_estimators= 100)
xgb_c.fit(X_train,y_train)
y_pred_x = xgb_c.predict(X_test)
print(classification_report(y_test,y_pred_x,target_names=['benign', 'defacement','phishing','malware']))

score = metrics.accuracy_score(y_test, y_pred_x)
print("accuracy: %0.3f" % score)


# In[53]:


cm = confusion_matrix(y_test, y_pred_rf)
cm_df = pd.DataFrame(cm,
                    index = ['benign', 'defacement','phishing','malware'], 
                    columns = ['benign', 'defacement','phishing','malware'])
plt.figure(figsize=(8,6))	
sns.heatmap(cm_df, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


# In[54]:


feat_importance = pd.Series(rf.feature_importances_, index=X_train.columns)
feat_importance.sort_values().plot(kind="barh",figsize=(10,6))


# PREDICTION

# In[50]:


def main(url):
    
    status = []
    
    status.append(having_ip_address(url))
    status.append(abnormal_url(url))
    status.append(count_dot(url))
    status.append(count_www(url))
    status.append(count_atrate(url))
    status.append(no_of_dir(url))
    status.append(no_of_embed(url))
    
    status.append(shortening_service(url))
    status.append(count_https(url))
    status.append(count_http(url))
    
    status.append(count_per(url))
    status.append(count_ques(url))
    status.append(count_hyphen(url))
    status.append(count_equal(url))
    
    status.append(url_length(url))
    status.append(hostname_length(url))
    status.append(suspicious_words(url))
    status.append(digit_count(url))
    status.append(letter_count(url))
    status.append(fd_length(url))
    tld = get_tld(url,fail_silently=True)
      
    status.append(tld_length(tld))

    return status


# In[51]:


def get_prediction_from_url(test_url):
    features_test = main(test_url)
    #Due to update to scikit-learn, we now need a 2D array as a parameter to the predict function.
    features_test = np.array(features_test).reshape((1, -1))

    pred = lgb.predict(features_test)
    if int(pred[0]) == 0:
        
        res="SAFE"
        return res
    elif int(pred[0]) == 1.0:
        
        res="DEFACEMENT"
        return res
    elif int(pred[0]) == 2.0:
        res="PHISHING"
        return res
        
    elif int(pred[0]) == 3.0:
        
        res="MALWARE"
        return res


# In[55]:


urls = [''] #give user input
for url in urls:
     print(get_prediction_from_url(url))

