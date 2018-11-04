from __future__ import print_function, division
import os
import re
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.functional import sigmoid
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
sns.set_style("darkgrid")
plt.ion()


import warnings
warnings.filterwarnings("ignore")

def is_number(s):
    try:
        return int(s)
    except ValueError:
        return 43201

MONTHS = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,
          'May':5,'Jun':6,'Jul':7,'Aug':8,
          'Sep':9,'Oct':10,'Nov':11,'Dec':12}
def months(x):
    """ returns months since 1970"""
    x = x.split('-')[1:]
    m1 = MONTHS[x[0]]
    return 12*(int(x[1])-1970) + m1

def unix_time_months(x):
    m1 = x/(3600*24*30)

def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

class RatingsDataset(Dataset):
    def __init__(self, ratings, users, movies):
        self.ratings = ratings
        self.users   = users
        self.movies  = movies
        self.length  = len(ratings)

    def __len__(self):
        return self.length

    def encode_user(self,userId):
        # create the user input vector
        u = self.users.iloc[userId-1,:]
        return torch.tensor(u,dtype=torch.float)

    def encode_movie(self,movieId):
        # create the movie input vector
        m = self.movies.iloc[movieId-1,:]
        return torch.tensor(m,dtype=torch.float)

    def __getitem__(self, idx):
        data = self.ratings.iloc[idx,:]
        
        user_input = self.encode_user(int(data['userId']))
        movie_input = self.encode_movie(int(data['movieId']))
        rating = torch.tensor(data['rating'],dtype=torch.float)
        time = torch.tensor(data['time'],dtype=torch.float)

        sample = {'user': user_input, 
                  'movie':movie_input, 
                  'time': time, 
                  'rating': rating}
        return sample

class NeuralRecommender(torch.nn.Module):
    def __init__(self,embedding_dim,n_users,n_movies,user_size,movie_size,
                 nn_sizes): 
        """
        inputs:
        -------
        n_users, n_movies: no. of different users and movies
        embedding_dim    : embedding dimensions for user and movie id
        user_size, movie_size: length of user/movie input 
        nn_sizes: length 2 tuple (n1,n2) where n1 is length of output of 
                  linear1 and n2 is the length of output of linear2
        """

        super(NeuralRecommender, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_users = n_users
        self.n_movies = n_movies

        # embedding layer
        self.user_embedding  = torch.nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = torch.nn.Embedding(n_movies, embedding_dim)

        # deep network
        self.input_size = 2*embedding_dim + user_size-1 + movie_size-1
        self.linear1 = torch.nn.Linear(self.input_size,nn_sizes[0])
        self.linear2 = torch.nn.Linear(nn_sizes[0],nn_sizes[1])
        self.linear3 = torch.nn.Linear(nn_sizes[1],1)
        
    def forward(self, user,movie):
        # matrix-factorization layer
        userId = user[:,0].long()
        movieId = movie[:,0].long()
        
        user_emb  = self.user_embedding(userId) 
        movie_emb = self.movie_embedding(movieId)
        y_MF = torch.mul(user_emb,movie_emb).sum(dim=1)

        # deep neural network
        nn_input = torch.cat([user_emb,user[:,1:],movie_emb,movie[:,1:]],dim=1)
        hidden = self.linear2(self.linear1(nn_input).clamp(min=0)).clamp(min=0)
        y_NN = self.linear3(hidden).squeeze()

        # combine the wide and deep-network outputs
        yhat = sigmoid(y_MF + y_NN)

        return yhat
