# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:35:34 2019

@author: delc7
"""

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %%

#Read the data scraped from https://millercenter.org/the-presidency/presidential-speeches by ND_DH_Project_Scrape_Data.py
presidents = pd.read_csv('presidents.csv', index_col = 0, parse_dates=['Birth Date', 'Death Date', 'Inauguration Date', 'Date Ended'])

#Remove all leading and trailing whitespace
presidents = presidents.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# %%
#Calculate age president was at inauguration
presidents['AgeInaugurated'] = (presidents['Inauguration Date'] - presidents['Birth Date']).astype('timedelta64[Y]')

#Calculate age president was at death
presidents['AgeDeath'] = (presidents['Death Date'] - presidents['Birth Date']).astype('timedelta64[Y]')

#Calulate years in office
presidents['YearsInOffice'] = (presidents['Date Ended'] - presidents['Inauguration Date']).astype('timedelta64[Y]')

# %%
#Manually fill in some presidents names:
fillNames = {'tyler': 'John Tyler', 'fillmore': 'Millard Fillmore', 'pierce': 'Franklin Pierce', 
             'buchanan': 'James Buchanan', 'lincoln': 'Abraham Lincoln', 'johnson': 'Andrew Johnson',
             'garfield': 'James A. Garfield', 'arthur': 'Chester A. Arthur', 'bharrison': 'Benjamin Harrison',
             'mckinley': 'William McKinley', 'roosevelt': 'Theodore Roosevelt', 'wilson': 'Woodrow Wilson',
             'coolidge': 'Calvin Coolidge', 'hoover': 'Herbert Hoover', 'truman': 'Harry Truman'}
for pres in presidents.index:
    if pd.isna(presidents.loc[pres, 'Full Name']):
        presidents.loc[pres, 'Full Name'] = fillNames[pres]
        
# %%

#Initialize some new binary columns
presidents['Lawyer'] = None
presidents['Military'] = None
presidents['Teacher/Professor'] = None

presidents['Democratic-Republican'] = None
presidents['Whig'] = None
presidents['Republican'] = None
presidents['Democrat'] = None

#Give some better defined attributes to presidents
for pres in presidents.index:
    career = presidents.loc[pres, 'Career'].lower()
    party = presidents.loc[pres, 'Political Party'].lower()
    ####   Careers ####
    #Can have had more than one career, so we don't use if else structure
    if 'lawyer' in career:
        presidents.loc[pres, 'Lawyer'] = 1
    else: 
        presidents.loc[pres, 'Lawyer'] = 0
    
    if 'officer' in career or 'soldier' in career:
        presidents.loc[pres, 'Military'] = 1
    else: 
        presidents.loc[pres, 'Military'] = 0
        
    if 'teacher' in career or 'professor' in career:
        presidents.loc[pres, 'Teacher/Professor'] = 1
    else: 
        presidents.loc[pres, 'Teacher/Professor'] = 0
        
    ####    Political Party ######
    if 'democratic-republican' in party:
        presidents.loc[pres, 'Democratic-Republican'] = 1
        #Check if president party contained democratic-republican when checking for democrat and republican
        drCheck = 1
    else:
        presidents.loc[pres, 'Democratic-Republican'] = 0
        drCheck = 0
    
    if 'whig' in party:
        presidents.loc[pres, 'Whig'] = 1
    else:
        presidents.loc[pres, 'Whig'] = 0

    if 'republican' in party and not drCheck:
        presidents.loc[pres, 'Republican'] = 1
    else:
        presidents.loc[pres, 'Republican'] = 0    
        
    if 'democrat' in party and not drCheck:
        presidents.loc[pres, 'Democrat'] = 1
    else:
        presidents.loc[pres, 'Democrat'] = 0   
        
        
# %%
summarystats = presidents.describe(include = [np.number])
sumRow = {}
for col in presidents.select_dtypes(include = [np.number]).columns:
    sumRow[col] = np.nansum(presidents[col])
rowNames = summarystats.index
rowNames = rowNames.append(pd.Index(['sum']))
summarystats = summarystats.append(sumRow, ignore_index = True)
summarystats.index = rowNames

print(summarystats)
# %%
#Some plots
#Subset to the 35 presidents who are either republican or democrats
presSimple = presidents.where((presidents['Democrat'] == 1) | (presidents['Republican'] == 1)).dropna(0,'all')

g = sns.FacetGrid(presSimple, row = 'Lawyer', col = 'Democrat', ylim = (0,4))
g.map(plt.hist, "AgeInaugurated")
plt.show()

g = sns.FacetGrid(presSimple, row = 'Lawyer', col = 'Democrat', ylim = (0,4))
g.map(plt.hist, "AgeDeath")
plt.show()

g = sns.FacetGrid(presSimple, row = 'Lawyer', col = 'Democrat')
g.map(plt.hist, "YearsInOffice")
plt.show()
#Note: Opposite nature of AgeInauguration and AgeDeath for Democratic Lawyers


sns.boxplot(data = presSimple, x = "Democrat", y = "AgeInaugurated")
plt.show()

sns.boxplot(data = presSimple, x = "Democrat", y = "AgeDeath")
plt.show()

sns.boxplot(data = presSimple, x = "Democrat", y = "YearsInOffice")
plt.show()





