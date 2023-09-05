### KICKOFF - CODING AN APP IN STREAMLIT

### import libraries
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import sys
import pandas as pd
import numpy as np
import requests
from spotify_dl import spotify_dl
from pathlib import Path
import time
import os
from dotenv import load_dotenv  # changed magic command to explicit load
import librosa
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import pairwise
from sklearn.model_selection import train_test_split
from typing import List
from flask import Flask, redirect, request
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping
from skimage.transform import resize
import streamlit as st
import joblib


pd.set_option("display.max_rows", None)  # pandas dataframe formatting options
pd.set_option("display.max_columns", None)


custom_env_path = "../../brainstation_capstone_cfg.env"  # environment variables file

# Spotify Developer Credentials
load_dotenv(dotenv_path=custom_env_path)
CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
# client ID from app
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")
# client secret from app
REDIRECT_URI = os.environ.get("REDIRECT_URI")
# redirect URI - the URI used here matches the one used within the app
SCOPE = "{} {}".format(os.environ.get("SCOPE_PUBLIC"), os.environ.get("SCOPE_PRIVATE"))
# formatted the scope this way to allow for custom configurations in the future
USERNAME = os.environ.get("SPOTIFY_USERNAME")
# Spotify username

# Read In Data
@st.cache
def load_data():
    training_mp3s_df = pd.read_parquet('../data/20230905_training_mp3s.parquet')
    normal_df = pd.read_parquet(
        "../data/vectorized_mp3s/pairwise_complete_parquets/20230901_complete_pairwise_data.parquet"
    )
    cnn_df = pd.read_parquet(
        "../data/vectorized_mp3s/cnn_complete_parquets/20230901_complete_cnn_data.parquet"
    )
    normal_similarity = pairwise.cosine_similarity(normal_df, dense_output=True)
    normal_similarity_sorted = np.argsort(normal_similarity)[:, ::-1]
    cnn_similarity = pairwise.cosine_similarity(cnn_df, dense_output=True)
    cnn_similarity_sorted = np.argsort(cnn_similarity)[:, ::-1]

    return training_mp3s_df, normal_df, normal_similarity, normal_similarity_sorted, cnn_df, cnn_similarity, cnn_similarity_sorted

def playlist_generator(normal_recommendations, cnn_recommendations):
    # Initialize SpotifyOAuth
    sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    cache_path=".cache",
    show_dialog=False,
    )
    # Get cached token or start authorization process
    access_token_info = sp_oauth.get_cached_token()
    if access_token_info:
        access_token = access_token_info["access_token"]
    else:
        print("Authorization required. Visit the following URL to authorize:")
        auth_url = sp_oauth.get_authorize_url()
        print(auth_url)
        return

    # Initialize spotipy with access token
    sp = spotipy.Spotify(auth=access_token)

    # Replace 'Your Username' with your Spotify username
    username = USERNAME
    for track_recommendation in normal_recommendations:
        playlist_name = (
            f"Pairwise Recommendation Playlist for Track {track_recommendation[0]}"
        )
        playlist_description = f"These are the top 10 pairwise matches for track {track_recommendation[0]} by cosine similarity in descending order."

        try:
            # Create a blank playlist
            playlist = sp.user_playlist_create(
                user=username,
                name=playlist_name,
                description=playlist_description,
                public=False,
            )
            print(f"Pairwise Playlist '{playlist_name}' was created")
        except spotipy.SpotifyException as e:
            print(f"An error occurred: {e}")

        # List of track IDs (URIs) to add to the playlist
        prefix = "spotify:track:"
        track_uris = list(prefix + x for x in track_recommendation)

        # Add tracks to the playlist
        sp.user_playlist_add_tracks(
            user=username, playlist_id=playlist["id"], tracks=track_uris
        )
    for track_recommendation in cnn_recommendations:
        playlist_name = (
            f"CNN Recommendation Playlist for Track {track_recommendation[0]}"
        )
        playlist_description = f"These are the top 10 CNN matches for track {track_recommendation[0]} by cosine similarity in descending order."

        try:
            # Create a blank playlist
            playlist = sp.user_playlist_create(
                user=username,
                name=playlist_name,
                description=playlist_description,
                public=False,
            )
            print(f"CNN Playlist '{playlist_name}' was created")
        except spotipy.SpotifyException as e:
            print(f"An error occurred: {e}")

        # List of track IDs (URIs) to add to the playlist
        prefix = "spotify:track:"
        track_uris = list(prefix + x for x in track_recommendation)

        # Add tracks to the playlist
        sp.user_playlist_add_tracks(
            user=username, playlist_id=playlist["id"], tracks=track_uris
        )

training_mp3s_df, normal_df, normal_similarity, normal_similarity_sorted, cnn_df, cnn_similarity, cnn_similarity_sorted = load_data()
np.random.seed(123)
rows = np.random.choice(11578, 10, replace=False)
normal_recommendations,cnn_recommendations = [],[]
for row in rows:
    norm_vec = list(normal_similarity_sorted[row, 0:11])
    normal_recommendation_vec = list(normal_df.iloc[norm_vec, :].index)
    normal_recommendations.append(normal_recommendation_vec)
    cnn_vec = list(cnn_similarity_sorted[row, 0:11])
    cnn_recommendation_vec = list(cnn_df.iloc[cnn_vec, :].index)
    cnn_recommendations.append(cnn_recommendation_vec)

playlist_generator(normal_recommendations, cnn_recommendations)

# get a dictinary of full paths to track hashes
# could then display pandas dataframe of the song and it's recommendations
# dictionary that maps hashes to row indices
# row_to_hash and hash_to_row dictionary
st.write('Streamlit is an open-source app framework for Machine Learning and Data Science teams. For the docs, please click [here](https://docs.streamlit.io/en/stable/api.html). \
    This is is just a very small window into its capabilities.')


#######################################################################################################################################
### LAUNCHING THE APP ON THE LOCAL MACHINE
### 1. Save your *.py file (the file and the dataset should be in the same folder)
### 2. Open git bash (Windows) or Terminal (MAC) and navigate (cd) to the folder containing the *.py and *.csv files
### 3. Execute... streamlit run <name_of_file.py>
### 4. The app will launch in your browser. A 'Rerun' button will appear every time you SAVE an update in the *.py file


#######################################################################################################################################
### Create a title

st.title("Kickoff - Live coding an app")

# Press R in the app to refresh after changing the code and saving here

### You can try each method by uncommenting each of the lines of code in this section in turn and rerunning the app

### You can also use markdown syntax.
#st.write('# Our last morning kick off :sob:')

### To position text and color, you can use html syntax
#st.markdown("<h1 style='text-align: center; color: blue;'>Our last morning kick off</h1>", unsafe_allow_html=True)


#######################################################################################################################################
### DATA LOADING

### A. define function to load data
@st.cache # <- add decorators after tried running the load multiple times
def load_data(path, num_rows):

    df = pd.read_csv(path, nrows=num_rows)

    # Streamlit will only recognize 'latitude' or 'lat', 'longitude' or 'lon', as coordinates

    df = df.rename(columns={'Start Station Latitude': 'lat', 'Start Station Longitude': 'lon'})     
    df['Start Time'] = pd.to_datetime(df['Start Time'])      # reset dtype for column
     
    return df

### B. Load first 50K rows
df = load_data("NYC_bikes_small.csv", 50000)

### C. Display the dataframe in the app
st.dataframe(df)


#######################################################################################################################################
### STATION MAP

st.subheader('Location Map - NYC bike stations')      

st.map(df)     


#######################################################################################################################################
### DATA ANALYSIS & VISUALIZATION

### B. Add filter on side bar after initial bar chart constructed

st.sidebar.subheader("Usage filters")
round_trip = st.sidebar.checkbox('Round trips only')

if round_trip:
    df = df[df['Start Station ID'] == df['End Station ID']]


### A. Add a bar chart of usage per hour

st.subheader("Daily usage per hour")

counts = df["Start Time"].dt.hour.value_counts()
st.bar_chart(counts)

### The features we have used here are very basic. Most Python libraries can be imported as in Jupyter Notebook so the possibilities are vast.
#### Visualizations can be rendered using matplotlib, seaborn, plotly etc.
#### Models can be imported using *.pkl files (or similar) so predictions, classifications etc can be done within the app using previously optimized models
#### Automating processes and handling real-time data


#######################################################################################################################################
### MODEL INFERENCE

st.subheader("Using pretrained models with user input")

# A. Load the model using joblib
model = joblib.load('sentiment_pipeline.pkl')

# B. Set up input field
text = st.text_input('Enter your review text below', 'Best. Restaurant. Ever.')

# C. Use the model to predict sentiment & write result
prediction = model.predict({text})

if prediction == 1:
    st.write('We predict that this is a positive review!')
else:
    st.write('We predict that this is a negative review!')

#######################################################################################################################################
### Streamlit Advantages and Disadvantages
    
st.subheader("Streamlit Advantages and Disadvantages")
st.write('**Advantages**')
st.write(' - Easy, Intuitive, Pythonic')
st.write(' - Free!')
st.write(' - Requires no knowledge of front end languages')
st.write('**Disadvantages**')
st.write(' - Apps all look the same')
st.write(' - Not very customizable')
st.write(' - A little slow. Not good for MLOps, therefore not scalable')
