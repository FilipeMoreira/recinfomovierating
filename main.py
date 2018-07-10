import numpy as np
import io
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from math import sqrt

def main():

    movie_limit = 100

    movie_names = []
    movie_count = 0
    movie_ratings = []
    users = []

    # reading movie title
    f = open('../movie_titles.txt', 'r', encoding = "ISO-8859-1")
    for line in f:
        if movie_count >= 10:
            continue
        values = line[:-1].split(',')
        movie_names.append(values[2])
        movie_count += 1

    print(str(movie_count) + " movies computed")

    # reading ratings
    for i in range(movie_limit):
        ratings = getRatingsForMovie(i+1)
        movie_ratings.append(ratings)
        # acquiring users
        for user_id in ratings.keys():
            if users.count(user_id) < 1:
                users.append(int(user_id))

    users.sort()
    print(str(len(users)) + " users computed")

    # fun begins
    data_matrix = np.zeros((len(users), movie_limit))
    for i in range(movie_limit):
        for key in movie_ratings[i].keys():
            data_matrix[users.index(int(key))][i] = int(movie_ratings[i].get(key))

    # for prediction purposes we will try to remove the last user's first non null rating to try to
    # predict that value
    ground_value = 0
    ground_index = 0
    for i in range(movie_limit):
        if data_matrix[len(data_matrix)-1][i] != 0 and ground_index == 0:
            ground_value = data_matrix[len(data_matrix)-1][i]
            ground_index = i
            data_matrix[len(data_matrix)-1][i] = 0
    
    user_similarity = pairwise_distances(data_matrix, metric='cosine')
    movie_similarity = pairwise_distances(data_matrix.T, metric='cosine')

    pred = predict(data_matrix, user_similarity, type='user')

    predicted_value = pred[len(pred)-1][ground_index]

    #prediction_rmse = rmse(predicted_value, ground_value)

    print("Prediction value: " + str(predicted_value))
    print("Real value: " + str(ground_value))
    #print("RMSE: " + prediction_rmse)

def getRatingsForMovie(movie_id):
    ratings = {}
    zeros = '0' * (7 - len(str(movie_id)))
    f = open('../training_set/mv_' + zeros + str(movie_id) + '.txt', 'r', encoding = "ISO-8859-1")
    for line in f:
        if len(line[:-1].split(',')) < 2:
            continue
        values = line[:-1].split(',')
        ratings[values[0]] = values[1]
    return ratings

def rmse(prediction, ground_truth):
    return sqrt(mean_squared_error(prediction, ground_truth))

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'movie':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

main()