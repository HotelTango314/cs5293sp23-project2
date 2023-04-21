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
    #get the data
    with open('docs/yummly.json','r') as f:
        recipes = json.load(f)
    rec_df = pd.DataFrame.from_dict(pd.json_normalize(recipes),orient='columns')
    X = []
    y = []
    labs = []
    for x in rec_df['ingredients']:
        X.append(list_to_sent(x))
    for z in rec_df['cuisine']:
        y.append(z)
    for a in rec_df['id']:
        labs.append(a)

    #vectorize the data
    vectorizer = TfidfVectorizer(max_df = .75, min_df = 5, stop_words="english")
    X_tfidf = vectorizer.fit_transform(X)
    
    #reduce the data
    lsa = make_pipeline(TruncatedSVD(n_components = 100), Normalizer(copy=False))
    X_lsa = lsa.fit_transform(X_tfidf)
    kmeans = KMeans(n_clusters=20, max_iter=100, n_init=1)
    kmeans.fit(X_lsa)
    center_pts = kmeans.cluster_centers_
    
    #label the clusters, majority rules
    cluster_label = [[] for _ in range(20)]
    cluster_dict = dict()
    for x in range(len(kmeans.labels_)):
        cluster_label[kmeans.labels_[x]].append(y[x])
    for x in range(len(cluster_label)):
        c = Counter(cluster_label[x])
        value, count = c.most_common()[0]
        cluster_dict[x] = value
    
    #predict new data
    new_recipe = [list_to_sent(ingredient)]
    new_point = lsa.transform(vectorizer.transform(new_recipe))
    pred_clust = kmeans.predict(new_point)
    
    #distance from new point to centroid of assigned cluster
    cent_dist = np.linalg.norm(new_point - center_pts[pred_clust])

    #distance from new point to N closest neighbors
    pts_in_clust = list((np.array(kmeans.labels_)==pred_clust[0]).nonzero()[0])
    clust_pts = X_lsa[pts_in_clust]
    clust_id = np.array(labs)[pts_in_clust]
    
    distances = []
    for x in clust_pts:
        distances.append(np.linalg.norm(new_point - x))
    idx = np.argsort(distances)[:(int(N))]
    distances = np.array(distances)[idx]
    labels = np.array(clust_id)[idx]

    #FINAL DATA
    cuisine = cluster_dict[pred_clust[0]]
    cuisine_dist = cent_dist
    nbr_dist = []
    for x,y in zip(labels, distances):
        nbr_dist.append({'id':x,'score':y})
    final_dict = {'cuisine':cuisine, 'score':cuisine_dist,'closest':nbr_dist}
    print(final_dict)
    print(json.dumps(final_dict, sort_keys=False,indent=4, separators=(',',': ')))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--N', required=True, help="Provide number of neighbors")
    parser.add_argument('--ingredient', action='append')

    args = parser.parse_args()
    recommender(args.N, args.ingredient)
