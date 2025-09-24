
#Generate searchabr ourput
from search_bar import *

search_bar = search_bar('Formula 1', allow_options = True) # the options output can be shown  
search_bar = search_bar('Formula 1, Racing', allow_options = False) 


#Generate recommendation output 
from recommendation import *

#only one output json
recommendation = generate_recommendation('home office', prompt_version='v02')


#List with the recommendation outputs for all subcategories
import pandas as pd
categories_file = 'Categories.xlsx'
df = pd.read_excel(categories_file)
subcategories = df['Subcategory'].dropna().tolist()

recommendations = []

for subcat in subcategories[:2]:

    recommendations.append(generate_recommendation(subcat, prompt_version='v02'))



# generate set of 5 trendning topics 
from trending import *

trending = generate_trending_set() #set of 5 trending topics





#geberate set of 3 update me news 

from update_me import *
set = generate_update_me_news_set() # news set, 3 news, list of dictionaries



#generate output for each topic 

categories_file = 'Categories.xlsx'
df = pd.read_excel(categories_file)
subcategories = df['Subcategory'].dropna().tolist()

update_me = []

update_me = []

for subtopic in subcategories[:2]:  #remove the index 2 if need to generate the update_me elemts for all topics 
        update_me.append(generate_update_me_element(keywords = subtopic, 
                                                    prompt_type = 'update_me_general', 
                                                    categories_file = categories_file,  
                                                    prompt_version='v01'))     

