import numpy
import io
from sklearn.metrics import mean_squared_error
from math import sqrt

def main():

    movie_limit = 10

    movie_names = []
    movie_count = 0

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



def getRatingsForMovie(movie_id):
    ratings = []
    zeros = '0' * (7 - len(srt(movie_id)))
    f = open('../training_set/mv' + zeros + movie_id + '.txt', 'r', encoding = "ISO-8859-1")
    for line in f:
        values = line[:-1].split(',')
        rating = { 'user_id': values[0], 'rate': values[1] }
        ratings.append(rating)
    return ratings

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

main()