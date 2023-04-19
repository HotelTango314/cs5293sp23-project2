import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import argparse
import pandas as pd
import json
from collections import Counter

def list_to_sent(lst):
    text = ''
    for i in lst:
        text = text + i + ', '
    return text


def recommender(N, ingredient):
    with open('docs/yummly.json','r') as f:
        recipes = json.load(f)
    rec_df = pd.DataFrame.from_dict(pd.json_normalize(recipes),orient='columns')
    X = []
    y = []
    for x in rec_df['ingredients']:
        X.append(list_to_sent(x))
    for z in rec_df['cuisine']:
        y.append(z)
    
    vectorizer = TfidfVectorizer(max_df = .75, min_df = 5, stop_words="english")
    X_tfidf = vectorizer.fit_transform(X)
    
    lsa = make_pipeline(TruncatedSVD(n_components = 100), Normalizer(copy=False))
    X_lsa = lsa.fit_transform(X_tfidf)
    kmeans = KMeans(n_clusters=20, max_iter=100, n_init=1)
    kmeans.fit(X_lsa)
    
    cluster_label = [[] for _ in range(20)]
    cluster_dict = dict()
    for x in range(len(kmeans.labels_)):
        cluster_label[kmeans.labels_[x]].append(y[x])
    for x in range(len(cluster_label)):
        c = Counter(cluster_label[x])
        value, count = c.most_common()[0]
        cluster_dict[x] = value
    


    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--N', required=True, help="Provide number of neighbors")
    parser.add_argument('--ingredient', action='append')

    args = parser.parse_args()
    recommender(args.N, args.ingredient)
