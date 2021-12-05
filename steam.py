# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 14:16:25 2021

@author: ningz
"""

import numpy as np
import pandas as pd
import altair as alt
import sklearn
import streamlit as st
import fsspec

st.title("Steam Games")

df = pd.read_csv("C:Users//ningz//Downloads//steam.csv")
df

st.markdown("**In the dataset, the first topic we are going to explore is: Who develops the most games in steam so far? **")
guess = st.selectbox("Do you want to make a guess? I'll give you a few choices, pick one and check your answer!",["KOEI TECMO GAMES CO., LTD.","Nikita Ghost_RUS","Adept Studios GD","Choice of Games","Valve"])
dev = df["developer"].value_counts().idxmax()
if guess == dev:
    st.write("You did it! The one who develops the most games is Choice of Games!")
else:
    st.write("I'm sorry it's not the correct answer but no worries! The correct answer is: Choice of Games!")

st.markdown("**Before we go straight into next topic, You can take a look at the release time distribution of the most recent 5000 games(as of the date of dataset creation) on the left!**")

rel = alt.Chart(df.iloc[:5000]).mark_bar().encode(
    y = "release_date",
    x = "count()"
)
st.sidebar.write(rel)

st.markdown("**Now, our second topic is to explore the proportion of games which are released on the first day of 2019.**")

new_year = df[(df["release_date"]=="2019-01-01")]
new_year

prob = len(new_year.iloc[:,0])/len(df.iloc[:,0])

st.write(f"About {100*prob} percent of games in this dataset are released on the first day of 2019.")
st.markdown("Let's pick a date by yourself and check the proportion!")

date = st.text_input("Write down your favourite date from 1997-06-29 to 2019-04-30, the format is ****-**-**")

pick = df[(df["release_date"]==date)]
pick

prob1 = len(pick.iloc[:,0])/len(df.iloc[:,0])
st.write(f"The proportion of games which are released on {date} is {prob1}!")

st.markdown("**Here I'm gonna check if there is a association between a game's positive ratings and average playtime. Since Altair can only include 5000 rows, I'm gonna create a dataframe which only contains the games which are released in 2019. We are gonna focus on this datatframe in this part.**")

df["release_year"] = df["release_date"].apply(lambda x: x[:4])
df["release_year"].astype(str).astype(int)
df = df.sort_values("release_year", ascending=False)
y2019 = df[(df["release_year"]=="2019")]
y2019

brush = alt.selection_interval(empty='none')

chart = alt.Chart(y2019).mark_circle().encode(
    x = "positive_ratings",
    y = "average_playtime",
    color = alt.condition(brush,
                          alt.Color("price:Q", scale=alt.Scale(scheme='turbo',reverse=True)),
                          alt.value("lightgrey")),
    tooltip = ["name"]
).add_selection(
    brush,
).properties(
    
    title="Steam Games"
)
st.write(chart)
st.markdown("It seems like there is no obvious association between a game's positive ratings and average playtime :( ")

st.markdown("**My last topic is to predict the owners range using the game's price and positive ratings. Before we build the model, let's standardize the data first. (Here we we go back to the original dataframe)**")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df1 = df[["price","positive_ratings"]]
scaler.fit(df1)
df1 = scaler.transform(df1)
df[["price","positive_ratings"]] = df1

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=10)

X1 = df[["price", "positive_ratings"]]
y1 = df["owners"]
clf.fit(X1,y1)

st.markdown("Our prediction is in the last column! As you can see, it's not completely accurate but it's almost! ")
df["pred"] = clf.predict(df[["price", "positive_ratings"]])
df

st.markdown("At last, let's make a prediction! Enter a price and a number of positive ratings below and get the owner range!")
price = st.text_input("Price", value="1")
pos_rt = st.text_input("Positive ratings", value="1")
A = scaler.transform([[price,pos_rt]])
pred = clf.predict(A)
pred
