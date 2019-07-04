# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:51:40 2019

@author: delc7
"""

# %%
import tfidfDist_V3 #program that takes care of vectorization

tfidf = tfidfDist_V3.tfidf #Vectorized values for each word in each doc
tfidf = tfidf[['tf_idf']]
tfidf.reset_index(inplace = True) 

files = tfidfDist_V3.files #Files. index matches tfidf.index.levels[0]

import ND_DH_Project_EDA
presidents = ND_DH_Project_EDA.presidents

import re
from statistics import mode
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras

import seaborn as sns
import matplotlib.pyplot as plt

# %% Functions

#Thanks to https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html for the confusion matrix plot with normalized
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def analyze_model(model, x, y, classes):
    y_pred = model.predict(x)
    #conf_matrix = confusion_matrix(y, y_pred)
    #print('Confusion matrix:')
    #print(conf_matrix)
    print('Accuracy:', accuracy_score(y, y_pred))
    if type(y[0]) == str:
        print('Guessing all most:', y.count(mode(y))/len(y))
    else:
        print('Guessing all same:', max(sum(y)/len(y), 1 - sum(y)/len(y)))
    print(classification_report(y, y_pred))
    plot_confusion_matrix(y, y_pred, classes)
    
# %%
#Get of presidents corresponding to files
pres_files = [re.sub('speeches/|_.*', '', f) for f in files]

#Make sure all pres_files are in the index of presidents
for f in pres_files:
    if f in presidents.index:
        pass
    else:
        print('bad')

yio = [presidents[pres_files[d]:pres_files[d]]['YearsInOffice'].values[0] for d in range(len(pres_files))]

# %% Split year of speech into bins

year_of_speech = [int(re.sub('.*(\d{4}).*', '\\1', f)) for f in files]
bins = int((max(year_of_speech) - min(year_of_speech)) / 40)
hist, bin_edges = np.histogram(year_of_speech, bins = bins)
print(hist)
print(bin_edges)
plt.hist(year_of_speech, bins = bins)
plt.title('Histogram of Year of Speech')
plt.show()

eras = ['era' + str(i+1) for i in range(bins)]
print('We have', bins, 'intervals. They are:', [eras[i] + ': ' + str(int(bin_edges[i])) + ' - ' + str(int(bin_edges[i+1])) for i in range(bins)])

speech_era = []
for i in range(len(year_of_speech)):
    y = year_of_speech[i]
    for e in range(bins):
        if y in range(int(bin_edges[e]),int(bin_edges[e+1])):
            speech_era.append(eras[e])
        elif y == 2018 and e == bins-1:
            speech_era.append(eras[-1])
                

# %%

#Transform to matrix style via pivot
tfidf = tfidf.pivot(index='Doc', columns='word', values='tf_idf')

#Convert NaN to 0
tfidf = tfidf.fillna(value = 0)

#Get lists of other info
dem = [presidents[pres_files[d]:pres_files[d]]['Democrat'].values[0] for d in range(len(pres_files))]
rep = [presidents[pres_files[d]:pres_files[d]]['Republican'].values[0] for d in range(len(pres_files))]
ageInaugurated = [presidents[pres_files[d]:pres_files[d]]['AgeInaugurated'].values[0] for d in range(len(pres_files))]
#normalize
ageInaugurated = [float(i)/max(ageInaugurated) for i in ageInaugurated]

#Put in vectorized data
#tfidf['dem_'] = dem
#tfidf['rep_'] = rep
#tfidf['ageIn_'] = ageInaugurated

# %% Additional cleaning

for name in tfidf.columns:
    if re.match('.*?\d.*?', name) or len(name)<3:
        #print("Dropping: ", name)
        tfidf=tfidf.drop([name], axis=1) ## do not use inplace=True. It will break for loop

# %%
yio_bin = []
for y in yio:
    if y <= 4:
        yio_bin.append(0)
    else:
        yio_bin.append(1)


#Split Trump
tfidf_trump = tfidf.loc[[i for i, x in enumerate(pres_files) if x == "trump"]]

tfidf_no_trump = tfidf.loc[[i for i, x in enumerate(pres_files) if x != "trump"]]
yio_no_trump = [yio[i] for i, x in enumerate(pres_files) if x != "trump"]
yio_bin_no_trump = [yio_bin[i] for i, x in enumerate(pres_files) if x != "trump"]

# %%






# %% TERM PREDICTIONS


sns.countplot(yio_bin_no_trump)
plt.xlabel('Longer than 4 Years')
plt.title('Counts of Speeches By Term Length')


# %% Term Length Split
    
#Split
X_train, X_test, y_train, y_test = train_test_split(tfidf_no_trump, yio_bin_no_trump, test_size = 0.3, random_state = 1000, stratify = yio_bin_no_trump)
classes = ['Years <= 4', 'Years > 4']

# %% Linear SVM to predict Term Length 3-Fold Cross Validation
    
svm_param_grid = {'C': [.3, 1, 4, 20, 50, 100, 200], 'max_iter': [100000]}

svm_gridsearch = GridSearchCV(cv = 3, estimator = LinearSVC(), param_grid = svm_param_grid, scoring = 'accuracy', n_jobs = -1, iid = False)
svm_gridsearch.fit(X_train, y_train)

print(svm_gridsearch.best_params_)
print(svm_gridsearch.best_score_)
analyze_model(svm_gridsearch, X_test, y_test, classes)

# %% Multinomial Naive Bayes

mnb_param_grid = {'alpha': [.000001, .0001, .01, .1, .5, 1, 2]}

mnb_gridsearch = GridSearchCV(cv = 3, estimator = MultinomialNB(), param_grid = mnb_param_grid, scoring = 'accuracy', n_jobs = -1, iid = False)
mnb_gridsearch.fit(X_train, y_train)

print(mnb_gridsearch.best_params_)
print(mnb_gridsearch.best_score_)
analyze_model(mnb_gridsearch, X_test, y_test, classes)

# %% Neural Network

#Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(200, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])


#Compile
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',#binary_crossentropy
              metrics=['accuracy'])

#Train
model.fit(X_train.values, np.array(y_train, ndmin=1), epochs = 40)

#Test
test_loss, test_acc = model.evaluate(X_test.values, np.array(y_test, ndmin=1))

print('Test accuracy:', test_acc)
print('Guessing all same:', max(sum(y_test)/len(y_test), 1 - sum(y_test)/len(y_test)))

print(classification_report(y_true = y_test, y_pred = np.argmax(model.predict(X_test), axis = 1)))

plot_confusion_matrix(y_true = y_test, y_pred = np.argmax(model.predict(X_test), axis = 1), classes = classes)
plt.show()

# %%
#Random Forest Model

random_forest = RandomForestClassifier(n_estimators=100)
rf_param_grid = {'max_depth': [4, 10, 35 ,70], 'max_features': ['auto', 'log2', 0.15, None]}

rf_gridsearch = GridSearchCV(cv = 3, estimator = random_forest, param_grid = rf_param_grid, scoring = 'accuracy', n_jobs = -1, iid = False)
rf_gridsearch.fit(X_train, y_train)

print(rf_gridsearch.best_params_)
print(rf_gridsearch.best_score_)
analyze_model(rf_gridsearch, X_test, y_test, classes)

# %% PARTY PREDICTIONS





# %% Party Split
    
#Split
#Remove non-dem/rep (Whigs, ...)
keep_index = [dem[i] == 1 or rep[i] == 1 for i in range(len(dem))]
keep_dem = [dem[i] for i in range(len(dem)) if keep_index[i]]
keep_tfidf = tfidf.loc[keep_index]
X_train, X_test, y_train, y_test = train_test_split(keep_tfidf, keep_dem, test_size = 0.3, random_state = 1000, stratify = keep_dem)
classes = ['Democrat', 'Republican']

sns.countplot(keep_dem)
plt.xlabel('Democrat')
plt.title('Counts of Democrats in Models')
plt.show()



# %% Linear SVM to predict Term Length 3-Fold Cross Validation
    
svm_param_grid = {'C': [.3, 1, 4, 20, 50, 100, 200], 'max_iter': [100000]}

svm_gridsearch = GridSearchCV(cv = 3, estimator = LinearSVC(), param_grid = svm_param_grid, scoring = 'accuracy', n_jobs = -1, iid = False)
svm_gridsearch.fit(X_train, y_train)

print(svm_gridsearch.best_params_)
print(svm_gridsearch.best_score_)
analyze_model(svm_gridsearch, X_test, y_test, classes)

# %% Multinomial Naive Bayes

mnb_param_grid = {'alpha': [.000001, .0001, .01, .1, .5, 1, 2]}

mnb_gridsearch = GridSearchCV(cv = 3, estimator = MultinomialNB(), param_grid = mnb_param_grid, scoring = 'accuracy', n_jobs = -1, iid = False)
mnb_gridsearch.fit(X_train, y_train)

print(mnb_gridsearch.best_params_)
print(mnb_gridsearch.best_score_)
analyze_model(mnb_gridsearch, X_test, y_test, classes)


# %% Neural Network

#Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])


#Compile
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',#binary_crossentropy
              metrics=['accuracy'])

#Train
model.fit(X_train.values, np.array(y_train, ndmin=1), epochs = 20)

#Test
test_loss, test_acc = model.evaluate(X_test.values, np.array(y_test, ndmin=1))

print('Test accuracy:', test_acc)
print('Guessing all same:', max(sum(y_test)/len(y_test), 1 - sum(y_test)/len(y_test)))

print(classification_report(y_true = y_test, y_pred = np.argmax(model.predict(X_test), axis = 1)))

plot_confusion_matrix(y_true = y_test, y_pred = np.argmax(model.predict(X_test), axis=1), classes = classes)
plt.show()

# %%
#Random Forest Model

random_forest = RandomForestClassifier(n_estimators=100)
rf_param_grid = {'max_depth': [4, 10, 35 ,70], 'max_features': ['auto', 'log2', 0.15, None]}

rf_gridsearch = GridSearchCV(cv = 3, estimator = random_forest, param_grid = rf_param_grid, scoring = 'accuracy', n_jobs = -1, iid = False)
rf_gridsearch.fit(X_train, y_train)

print(rf_gridsearch.best_params_)
print(rf_gridsearch.best_score_)
analyze_model(rf_gridsearch, X_test, y_test, classes)

# %% Year of speech



plt.hist(year_of_speech, bins = bins)
plt.title('Histogram of Year of Speech')
plt.show()




# %% Era Split
    
#Split
X_train, X_test, y_train, y_test = train_test_split(tfidf, speech_era, test_size = 0.3, random_state = 1000, stratify = speech_era)

classes = eras


# %% Linear SVM to predict Term Length 3-Fold Cross Validation
    
svm_param_grid = {'C': [.3, 1, 4, 20, 50, 100, 200], 'max_iter': [100000]}

svm_gridsearch = GridSearchCV(cv = 3, estimator = LinearSVC(), param_grid = svm_param_grid, scoring = 'accuracy', n_jobs = -1, iid = False)
svm_gridsearch.fit(X_train, y_train)

print(svm_gridsearch.best_params_)
print(svm_gridsearch.best_score_)
analyze_model(svm_gridsearch, X_test, y_test, classes)

# %% Multinomial Naive Bayes

mnb_param_grid = {'alpha': [.000001, .0001, .01, .1, .5, 1, 2]}

mnb_gridsearch = GridSearchCV(cv = 3, estimator = MultinomialNB(), param_grid = mnb_param_grid, scoring = 'accuracy', n_jobs = -1, iid = False)
mnb_gridsearch.fit(X_train, y_train)

print(mnb_gridsearch.best_params_)
print(mnb_gridsearch.best_score_)
analyze_model(mnb_gridsearch, X_test, y_test, classes)

# %% Neural Network

#Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(800, activation=tf.nn.relu),
    keras.layers.Dense(bins + 1, activation=tf.nn.softmax)
])


#Compile
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',#binary_crossentropy
              metrics=['accuracy'])

#Train
model.fit(X_train.values, np.array([re.findall(r'\d+', s) for s in y_train], ndmin=1), epochs = 45)

#Test
test_loss, test_acc = model.evaluate(X_test.values, np.array([re.findall(r'\d+', s) for s in y_test], ndmin=1))

print('Test accuracy:', test_acc)
print('Guessing all most:', y_test.count(mode(y_test))/len(y_test))

print(classification_report(y_true = y_test, y_pred = ['era' + str(np.argmax(model.predict(X_test), axis = 1)[i]) for i in range(len(y_test))]))

plot_confusion_matrix(y_true = np.array([int(re.findall(r'\d+', s)[0]) for s in y_test], ndmin=1), y_pred = np.argmax(model.predict(X_test), axis=1), classes = classes)
plt.show()

# %%
#Random Forest Model

random_forest = RandomForestClassifier(n_estimators=100)
rf_param_grid = {'max_depth': [4, 10, 35 ,70], 'max_features': ['auto', 'log2', 0.15, None]}

rf_gridsearch = GridSearchCV(cv = 3, estimator = random_forest, param_grid = rf_param_grid, scoring = 'accuracy', n_jobs = -1, iid = False)
rf_gridsearch.fit(X_train, y_train)

print(rf_gridsearch.best_params_)
print(rf_gridsearch.best_score_)
analyze_model(rf_gridsearch, X_test, y_test, classes)

# %%





