# cluster evaluation 
#  JMA sept 2023
#
# used to compute entropy in cluster evaluation notebook
# Note: entropy functions used in aa_cluster_demo have been moved to aa_cluster.py

"""Here we evaluate how well a particular clustering captures aspects 
of the text by using "aspect patterns" in the form of regular expressions 
that flag passages related to a particular aspect (which could be a topic, 
application domain, semantic framework, etc.). Clusterings that capture an 
aspect well will concentrate instances of that aspect in a smaller number of 
clusters than those that do not capture the aspect well, in which the aspect 
instances ahould be more randomly distrubuted. 

The quality of the clusters is measured by their purity. Purity is a labelled 
criterion for a binary label -- true or false -- for the cases in the cluster. 
As the fraction of labels reaches an extreme the purity increases.  As purity 
increases the entropy of the cluster frequency decreases.  We use entropy to 
quantify the degree of concentration of the aspect pattern across the clusters 
by comparing the average entropy of the clustering with the entropy of the 
unclustered cases.  Any clustering will decrease these two entropy measures. 
We use the delta entropy, the decrease in entropy (a positive number, more is 
better) to measure the clustering quality."""

import regex, math, os, sys
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Patterns are the same ones used in "Aspect-aware semantic clustering".

from text_patterns import patterns

# sys.path.append('misc/')
# from utils import utils

POS_CLUSTER_FILE = "../../data/dolly_pos_clusters.parquet"

# Entropy-based evaluation

def counts(col, pattern, df):
    return df[df['pattern']== pattern][col].sum()

def clusters(pattern, df):
    return df[df.pattern == pattern].index

def h(true_count, total_count):
    if total_count == 0:
        print("WARNING Computing entropy for zero counts. Returning 0")
        return 0.0
    else:
        p = float(true_count) / total_count
        #print(f'p: {p}')
    'Natural log based entropy'
    if (p < sys.float_info.epsilon) or 1-p < (sys.float_info.epsilon):
        return 0.0
    else:
        return -p * math.log(p) - (1-p) * math.log(1-p)

def mean_H(pattern, df):
    return h(counts(True, pattern, df), (counts(True, pattern, df) + counts(False, pattern, df)))

def avg_clusters_H(pattern, df):
    cluster_h_sum = []
    total_sum = []
    for cluster in clusters(pattern, df):
        dfc = df.loc[(df.index == cluster)]
        trues = counts(True, pattern, dfc)
        falses = counts(False, pattern, dfc)
        totals = trues + falses
        total_sum.append(totals)
        cluster_h_sum.append(totals * h(trues, totals))
    s = sum(cluster_h_sum)/(len(cluster_h_sum) * sum(total_sum))
    return s

def cluster_improvement_H(pattern, df):
    """Return a postive number; the decrease in entropy compared to no clustering. l
       Larger is better (e.g. the cluster's avg entropy is less than the unclustered entropy.)"""
    return mean_H(pattern, df) - avg_clusters_H(pattern, df)

def pattern_delta_entropy(df):
    pattern_Hs = []
    for pattern in np.unique(df.pattern):
        pattern_Hs.append({'pattern': pattern,
                        'mean_H':mean_H(pattern, df), 
                        'avg_clusters_H': avg_clusters_H(pattern, df), 
                        'cluster_improvement_H':cluster_improvement_H(pattern, df)})
    return pd.DataFrame(pattern_Hs)


###############################################################

def integerify(x):
    digits = regex.sub(r'[^0-9]', '', str(x))
    num = int(digits) if len(digits) > 0 else -1
    return num

def pattern_distribution_by_cluster(df, pattern_name, pattern_lookup, text_col, cluster_col):
    import re
    # integerify = lambda x: int(re.sub(r'[^0-9]', '', str(x))) # extract digits from variable as integer
    pattern = pattern_lookup[pattern_name]
    flags = [bool(regex.search(pattern, sent, regex.IGNORECASE)) for sent in df[text_col]]
    xtab = pd.crosstab(df[cluster_col], flags)
    xtab['pattern'] = pattern_name
    xtab['cluster_col'] = cluster_col
    xtab['frequency'] = [ row[True]/(row[True] + row[False]) for row in xtab.to_dict(orient='records') ]
    xtab['overall_count'] = np.sum(flags)
    xtab['overall_frequency'] = np.sum(flags)/len(flags)
    xtab['lift'] = [ row['frequency']/row['overall_frequency'] for row in xtab.to_dict(orient='records') ]
    xtab['cluster_number'] = [integerify(cluster_col) for i in xtab.index]
    return xtab

def get_all_pattern_distributions_by_cluster(df, pattern_lookup, text_col, cluster_col):
    pattern_distribution_pdf = pd.concat([pattern_distribution_by_cluster(df, pattern_name, pattern_lookup, text_col, cluster_col)
                for pattern_name in pattern_lookup.keys()], axis=0)
    # pattern_lift_pdf.columns = pattern_lookup.keys()
    # pldf = pattern_lift_pdf.reset_index(drop=False).rename(columns={cluster_col:'cluster'})
    return pattern_distribution_pdf

def load_patterns(pos_data, patterns):
    pos_pattern_distributions = get_all_pattern_distributions_by_cluster(pos_data, patterns, 'instruction', 'pos__B')
    return pos_pattern_distributions

if __name__ == '__main__':
    pos_data = pd.read_parquet(POS_CLUSTER_FILE, engine='pyarrow')
    pos_pattern_distributions = load_patterns(pos_data, patterns)
    print(pattern_delta_entropy(pos_pattern_distributions))