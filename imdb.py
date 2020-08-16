import numpy as np
import pandas as pd

metadata = pd.read_csv('movies_metadata.csv')
print(metadata.shape) # print shape DataFrame for reference

### Create a movie recommendation system
 # (1) develop a movie recommendation system based on rating
 # (2) develop a movie recommendation system based on content
### START with (1) develop an IMDB Top 250 movie list ###
# ensure all movies are rated equally because a 9 rating from 10 votes cannot
# be compared with an 8.8 from 10,0000 votes


C = metadata['vote_average'].mean()
print(C)
# add column to DataFrame for movies with vote counts in 90th percentile
# (i.e. the minimum number of votes to be in the recommendation system)
m = metadata['vote_count'].quantile(0.90)
print(m)

#create new filtered DataFrame of movies in 90th percentile
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
print(q_movies.shape) # print shape DataFrame for reference

# define a function to calculate the weighted rating of a movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m)*R) +(m/(m+v)*C) #formula source: https://help.imdb.com/article/imdb/track-movies-tv/ratings-faq/G67Y87TFYYP6TWAV# 

# create a new column to the DataFrame in which to define the weighted_rating
q_movies['score'] = q_movies.apply(weighted_rating, axis = 1)

# new DataFrame with titles and weighted rating
q_movies_top = q_movies[['title','score']].copy()
print(q_movies_top.shape)
# sort DataFrame on descending score
q_movies_top = q_movies_top.sort_values('score', ascending = False)

# create DataFrame with top 10 scoring movies (can be 250 too)
top_rated = q_movies_top.head(10)
print(top_rated)




### START with (2) develop a movie recommendation system based on content ###
 # develop recommendation system for movies with similar plot descriptions
 
metadata['overview'].head(5)
# create Term Frequency Inverse Document Frequency (TF-IDF) vector per
# movie description
# this creates a matrix where columns represent words (that appear =< 1)
# and each column a movie

# import TF-IDF vectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# define a TF-IDF vectorizer object removing all English stopwords (e.g. 'the')
tfidf = TfidfVectorizer(stop_words = 'english')

# replace not-a-number (NaN) overview cells with ''
metadata['overview'] = metadata['overview'].fillna('')

# construct the TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])
print(tfidf_matrix)

# now possible to compare how similar descriptions are by comparing their words
# for this compute cosine similarity of movies using linear_kernel
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# we are going to define a function taking the movie title as input and as
# output 10 recommended movies based on similar plot descriptions
# for this you need map of indices linked to movie titles
indices = pd.Series(metadata.index, index = metadata['title']).drop_duplicates()



def get_recommendations(title, cosine_sim = cosine_sim):
    idx = indices[title]
   # print(idx)
    sim_scores = list(enumerate(cosine_sim[idx]))
   # print(sim_scores [1:11])
    sim_scores = sorted(sim_scores, key = lambda x : x[1], reverse = True)  # sort movies based on similarity
    sim_scores = sim_scores[1:11]  #get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]  #get movie indices(titles)
   # print(movie_indices)
    return metadata['title'].iloc[movie_indices]

def get_recommendations_details(title):
    movie_list = []
    for i in get_recommendations(title).to_dict().keys():
        movie_list.append(i)       
    return metadata.loc[movie_list]


def get_10_recommendations(title):
    recommends_movies = get_recommendations_details(title)
    return get_recommendations(title), get_recommendations_details(title), recommends_movies


# now we define an input to the recommendation system (could use on a webiste)
print(get_10_recommendations(input('please provide a movie title: '))) #combine below two function


movie_title = input('please provide a movie title: ')
print(get_recommendations(movie_title)) #find 10 sim movies
print(get_recommendations_details(movie_title)) #get details of 10 sim movies
recommands_movie = get_recommendations_details(movie_title) #recommends_movie_dataframe
globals()[movie_title] = get_recommendations_details(movie_title) #create a recommends_movie_dataframe named as the moive input



##Example movies with direct output
get_recommendations('Fight Club')
get_recommendations('Jackass: The Movie')
get_recommendations('The Lord of the Rings: The Fellowship of the Ring')
get_recommendations('Good Will Hunting')
get_recommendations('The Big Short')
get_recommendations('A Beautiful Mind')

