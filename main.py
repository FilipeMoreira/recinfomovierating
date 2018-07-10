import numpy as np
import io
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from math import sqrt

def main():

    movie_limit = 10

    movie_names = []
    movie_count = 0
    movie_ratings = []
    users = []

    #reading movie title
    f = open('../movie_titles.txt', 'r', encoding = "ISO-8859-1")
    for line in f:
        if movie_count >= 10:
            continue
        values = line[:-1].split(',')
        movie_names.append(values[2])
        movie_count += 1

    print(movie_count)

    #reading ratings
    for i in range(movie_limit):
        ratings = getRatingsForMovie(i)
        movie_ratings.append(ratings)
        #acquiring users
        for user_id in ratings.keys():
            if users.count(user_id) < 1:
                users.append(user_id)

    users.sort()

    #fun begins
    data_matrix = np.zeros((len(users), movie_limit))
    for i in range(movie_limit):
        for key in movie_ratings[i].keys():
            data_matrix[key][i] = movie_ratings[i].get(key)

    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    movie_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

def getRatingsForMovie(movie_id):
    ratings = {}
    zeros = '0' * (7 - len(srt(movie_id)))
    f = open('../training_set/mv' + zeros + movie_id + '.txt', 'r', encoding = "ISO-8859-1")
    for line in f:
        values = line[:-1].split(',')
        ratings.[values[0]] = values[1]
    return ratings

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

main()