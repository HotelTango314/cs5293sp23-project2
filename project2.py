import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import json

def recommender(N, ingredient):
    with open('docs/yummly.json','r') as f:
        recipes = json.load(f)
    print(recipes['cuisine'])
    rec_df = pd.DataFrame.from_dict(pd.json_normalize(recipes),orient='columns')
    total_list = []
    for x in rec_df['ingredients']:
        for y in x:
            total_list.append(y)
    nodup = [*set(total_list)]
    for i in nodup:
        print(i)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--N', required=True, help="Provide number of neighbors")
    parser.add_argument('--ingredient', action='append')

    args = parser.parse_args()
    recommender(args.N, args.ingredient)
