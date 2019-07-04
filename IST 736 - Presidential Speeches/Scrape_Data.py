# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:21:39 2019

@author: delc7
"""
# %%
import urllib
from bs4 import BeautifulSoup as bs
import re
import pandas as pd

# %%


link = "https://millercenter.org/the-presidency/presidential-speeches"
f = urllib.request.urlopen(link)
myfile = f.read()
soup = bs(myfile, 'html.parser')
#print(soup.prettify())

# %%
#Get all li tags
li = soup.find_all('li')[4:46]

# %%
#Convert li tags to list of strings
links = []
for x in li:
    links.append(str(x))


# %%
#Get the link addition from the tags
pattern = re.compile(r'href=\"(\/.*?)\"')

links = [str('https://millercenter.org' + pattern.findall(lst)[0]) for lst in links]


# %%
presidents = {}
replaceInName = '/the-presidency/presidential-speeches/'
replaceInPresident = 'https://millercenter.org/president/'

for presLink in links:
    president = presLink.replace(replaceInPresident, '')
    presidents[president] = {}
    
    linkSpeeches = []
    names = []
    
    f = urllib.request.urlopen(presLink)
    myfile = f.read()
    #parse
    soup = bs(myfile, 'html.parser')
    #get the article section
    start = soup.find('article').find("div", {'class': 'speech-info-wrapper'})
    
    #Get the president info
    #Full Name, Birth Date, Death Date, Birth Place, Education, Religion, Career, Political Party, Nickname, Marriage, 
    #Children, Inauguration Date, Date Ended, President Number, Burial Place
    for label in soup.findAll('label'):
        if label.string in ['Full Name', 'Birth Date', 'Birth Place', 'Death Date', 'Burial Place', 
                            'Education', 'Religion', 'Career', 'Political Party', 'Nickname', 'Marriage', 
                            'Children', 'Inauguration Date', 'Date Ended', 'President Number']:
            nextInfo = label.findNext('div')
            while nextInfo.string is None:
                nextInfo = nextInfo.findNext()
            presidents[president][label.string] = nextInfo.string.replace(u'\u201c', '"').replace(u'\u201d', '"').replace(u'\u2015', '-').replace(u'\u2014', '-').replace(u'\u2013', '-').replace(u'\u2012', '-').replace(u'\u2019', "'")
    
    
    #find next three links
    try:
        name1 = pattern.findall(str(start.find_next('a')))[0]
        linkSpeech1 = str('https://millercenter.org' + name1)
        name1 = name1.replace(replaceInName, '')
        linkSpeeches.append(linkSpeech1)
        names.append(name1)
    except: pass
    
    try:
        name2 = pattern.findall(str(start.find_next('a').find_next('a')))[0]
        linkSpeech2 = str('https://millercenter.org' + name2)
        name2 = name2.replace(replaceInName, '')
        linkSpeeches.append(linkSpeech2)
        names.append(name2)
    except: pass
    
    try:
        name3 = pattern.findall(str(start.find_next('a').find_next('a').find_next('a')))[0]
        linkSpeech3 = str('https://millercenter.org' + name3)
        name3 = name3.replace(replaceInName, '')
        linkSpeeches.append(linkSpeech3)
        names.append(name3)
    except: pass
    
    
    for loc in range(3):
        #read the link to get the speech
        try:
            f = urllib.request.urlopen(linkSpeeches[loc])
            myfile = f.read()
            #parse
            soup = bs(myfile, 'html.parser')
            speech = soup.find("div", {'class': 'view-transcript'})
            newFile = 'speeches/' + president + '_' + names[loc] + '.txt'
            fout = open(newFile, 'w', encoding="utf-8")
            fout.write(str(speech))
            fout.close()
        except: pass



# %%
df = pd.DataFrame(columns = ['Full Name', 'Birth Date', 'Birth Place', 'Death Date', 'Burial Place', 
                            'Education', 'Religion', 'Career', 'Political Party', 'Nickname', 'Marriage', 
                            'Children', 'Inauguration Date', 'Date Ended', 'President Number'])

#Save information to a dataframe
for pres in presidents.keys():
    d = presidents[pres]
    tmp = pd.DataFrame(d, index = [pres,])
    df = df.append(tmp, sort = False)

df.to_csv('presidents.csv')

# %%




