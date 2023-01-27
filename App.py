import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import regex as re

from PIL import Image
import urllib
import json
import os
import math
import base64

#import modules and packages
import requests
from bs4 import BeautifulSoup
import json
import datetime


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Movie", page_icon=":clapper:", layout="wide")


with open( "style2.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)



@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("/Users/ligandrosy/Downloads/IMG.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1553377102-7479aacccd00?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

movies = pd.read_csv("/Users/ligandrosy/Downloads/movies.csv")

def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title


movies["clean_title"] = movies["title"].apply(clean_title)


vectorizer = TfidfVectorizer(ngram_range=(1,2)) # Toy,Toy Story,Story 1995 etc combos

tfidf = vectorizer.fit_transform(movies["clean_title"])

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    
    return results


# https://files.grouplens.org/datasets/movielens/ml-25m.zip
ratings = pd.read_csv("/Users/ligandrosy/Downloads/ratings.csv")



def find_similar_movies(movie_id,count):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(count).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]




col_list = movies["title"].values.tolist()

st.title(":clapper: Movie Recommender")
st.subheader("Made by : Ligandro")


x = st.selectbox("Enter Movie Name ",options = col_list)

count = 10
 
results = search(x)
movie_id = results.iloc[0]["movieId"]
df = find_similar_movies(movie_id,count)

st.table(df)

st.subheader("Highest the score , better the similiarity between movies")






