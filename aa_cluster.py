### Aspect aware clustering functions ###

import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def find_pattern_flags_in_text(text_list, pattern_dict):
    """
    Find each pattern in a list of text passages.
    text_list: a list of text passages.
    pattern_dict: key is the name of the pattern, value is a regular expression. 
            Note that we use the more sophisticated but backwardly compatible 'regex' package, rather than 're'
    returns: Pandas dataframe with one row per text passage and one column per regex pattern.

    example: pattern_flags = find_pattern_flags_in_text(instruction_data['instruction'], patterns)
        To count the number of each flag: pattern_flags.sum(axis=0)
    note: Does not preserve the index if you give it a Series instead of a list.
    """
    import regex
    import pandas as pd
    return pd.DataFrame.from_dict({
        pattern_name: [bool(regex.search(pattern_dict[pattern_name], sent, regex.IGNORECASE))
                           for sent in text_list]
        for pattern_name in pattern_dict
    })


def train_pattern_models(vector_list, flag_table, verbose=False):
    """
    vector_list: list of fixed length feature vectors (e.g., embeddings)
    flag_table: dataframe of binary labels; one row per vector and one column per label, ad produced by `find_pattern_flags_in_text`.
    pattern_dict: key is the name of the pattern, value is a regular expression. 
    Note: we use cross-validation and specify 'roc_auc' as the scoring metric 
        so we can pull cross-validation AUC values from the trained models.
    """
    import regex  # more sophisticated than 're', but backward compatible (handles lookarounds better)
    from sklearn.linear_model import LogisticRegressionCV
    
    models = {}
    # text_col = 'instruction'
    # vector_col = f"{text_col}_vector"
    X = [v for v in vector_list]
    for flag_col in flag_table.columns:
        if verbose:
            print(f"Fitting model for '{flag_col}'")
        
        clf = LogisticRegressionCV(cv=5, scoring='roc_auc', # fit_intercept=False, 
                                   n_jobs=-1, max_iter=10000)
        clf.fit(X, flag_table[flag_col])
        models[flag_col] = clf

    return models


def get_model_mean_xval_auc(models_dict):
    """
    show mean cross-validation AUC for each model.
    Note: These are LogisticRegressionCV models with scoring='roc_auc'.
    """
    from sklearn.linear_model import LogisticRegressionCV
    import numpy as np
    return {k: np.mean([np.max(v) for v in model.scores_[True]]) for k, model in models_dict.items()}


def get_model_scores_for_vectors(vectors_list, models_dict):
    import pandas as pd
    X = [v for v in vectors_list]
    return pd.DataFrame.from_dict({key: clf.predict_proba(X)[:,1] for key, clf in models_dict.items()})


def to_unit_vectors(v_list):
    """
    Scale each vector in the given list to have a length of 1.
    v_list: a list of pandas Series of vectors.
    returns: scaled list.
    """
    import numpy as np
    to_unit_vector = lambda v: v/np.linalg.norm(v)
    return [to_unit_vector(v) for v in v_list]


def get_projection(embedding, modifier):
    # v1 is the embedding, v2 is the modifier
    embedding = np.array(embedding) / np.linalg.norm(embedding)
    modifier = np.array(modifier) / np.linalg.norm(modifier)
    projection = np.dot(embedding, modifier.T) * embedding
    projection = projection.reshape(-1,)
    return projection


def get_cluster_dendrogram(vector_list, metric = 'cosine'):
    """
    returns: a linkage matrix representing a dendrogram [cl1, cl2, distance, size of merged cluster]. See `scipy.cluster.hierarchy` linkage.
    """
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import ward
    import pandas as pd

    if type(vector_list) == pd.core.series.Series:
        vector_list = vector_list.to_list()

    linkage_function = ward
    
    D = pdist(vector_list, metric=metric) # condensed pairwise distance matrix
    dendro = linkage_function(D) 
    return dendro


def get_reference_roc(flag_list, scores):
    """
    Compute ROC the old fashioned way to double cleck our custom 'get_cluster_roc' function. 
    This should give the same results as long as there are no clusters with tied scores.
    """
    # To Do: This function is only needed for testing; remove it once the custom method is thoroughly tested.
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(flag_list, scores)
    return pd.DataFrame.from_dict({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds})

def h(p):
    'Binary entropy using natural log'
    from math import log
    return 0.0 if min(p, 1-p) < sys.float_info.epsilon else -p * log(p) - (1-p) * log(1-p)

def get_cluster_stats(cluster_list, flag_list):
    """
    Compute frequency (p), entropy, and purity for each cluster.

    cluster_list: a list of cluster assignments (e.g., one column from the dataframe returned by 'get_cluster_assignments').
    flag_list: a list or Series of binary labels (e.g., one column from the dataframe returned by 'find_pattern_flags_in_text').
        These two lists must be or the same length and have corresponding elements.

    returns: a dataframe with one row per cluster and the following columns:
        'p': the fraction of positive items in the cluster
        'n': total number of items
        'pos': the number of positive items
        'neg': the number of negative items
        'entropy'
        'purity'
    """
    purity = lambda p: max(p, 1 - p)
    
    df = pd.DataFrame.from_dict({'cluster' : cluster_list, 'flag': flag_list})
    
    cluster_stats = df.groupby('cluster') \
           .agg(p=('flag', 'mean'), n=('flag', 'size'), pos=('flag', 'sum') )
    cluster_stats['neg'] = cluster_stats['n'] - cluster_stats['pos']
    cluster_stats['entropy'] = [ h(p) for p in cluster_stats['p'] ]
    cluster_stats['purity'] = [ purity(p) for p in cluster_stats['p'] ]
    
    return cluster_stats

def get_avg_clusters_H(df):
    cluster_h_sum = []
    total_sum = []
    for cluster in df.itertuples():
        trues = cluster.pos
        totals = cluster.n
        total_sum.append(totals)
        cluster_h_sum.append(totals * h(trues / totals))
    s = sum(cluster_h_sum)/(len(cluster_h_sum) * sum(total_sum))
    return s

def get_cluster_improvement_H(cluster_stats_df):
    """Return entropy delta for one set of clusters as a postive number; the decrease in entropy compared to no clustering. l
       Larger is better (e.g. the cluster's avg entropy is less than the unclustered entropy.)"""
    mean_h = cluster_stats_df[['pos', 'n']].sum()
    mean_entropy = h(mean_h.pos / mean_h.n) # - avg_clusters_H(pattern, df)
    improvement = mean_entropy - get_avg_clusters_H(cluster_stats_df)
    return improvement

def get_cluster_entropy_improvement(cstats):
	"""
	Return entropy delta for one set of clusters as a postive number; the decrease in entropy compared to no clustering. 
	Larger is better (e.g. the avg entropy for clusters is less than the unclustered entropy.)
	cstats: a cluster stats dataframe as returned by 'get_cluster_stats'
	"""
	overall_entropy = h( sum(cstats.pos) / sum(cstats.n) )
	mean_cluster_weighted_entropy = np.mean( (cstats.n * cstats.entropy)/sum(cstats.n) )
	return overall_entropy - mean_cluster_weighted_entropy


def get_cluster_roc(cluster_list, flag_list):
    """
    Compute ROC for clustering from scratch so that we will be one point per cluster, rather than one point per unique score.
    This will be particularly useful when there are a lot of singleton clusters with frequencies of exactly 1 or exactly 0.

    cluster_list: list with categorical cluster assignments.
    flag_list: list with binary labels. Positions in these two lists refer to the same items.
    returns: a dataframe with tpr and fpr (as well as some other bookkeeping columns).
    """
    
    df = get_cluster_stats(cluster_list, flag_list)

    zero_row = pd.DataFrame({'n': 0, 'p': float("inf"), 'pos': 0,'neg': 0}, index =[0])
    df = pd.concat([zero_row, df]).reset_index(drop = True)

    df = df.sort_values('p', ascending=False).reset_index(drop=True)
    df['pos_cumsum'] = np.cumsum(df['pos'])
    df['tpr'] = np.cumsum(df['pos'])/np.sum(df['pos'])
    df['fpr'] = np.cumsum(df['neg'])/np.sum(df['neg'])
    
    return df[ ['fpr', 'tpr', 'p'] ].rename(columns={'p': 'threshold'})


def get_performance_for_clustering(cluster_list, flag_list):
    """
    For the given cluster/flag pair, describe how well clusters predict binary labels (flags) using 
    frequency of positive labels in clusters 
        * compute an ROC curve (and AUC) using cluster label frequency as a score to predict the label
        * compute mean entropy and purity across clusters

    cluster_list: list of cluster IDs to which cases are assigned
    flag_list: binary label for cases
    """
    from sklearn import metrics

    cluster_stats = get_cluster_stats(cluster_list, flag_list)
    # cluster_improvement_H = get_cluster_improvement_H(cluster_stats)
    cluster_improvement_H = get_cluster_entropy_improvement(cluster_stats)
    
    scores = [ cluster_stats['p'][cluster_name] for cluster_name in cluster_list ]
    ref_roc_df = get_reference_roc(flag_list, scores)
    ref_auc = metrics.auc(ref_roc_df['fpr'], ref_roc_df['tpr']) # this will be true if there are no ties: np.allclose(ref_roc_df, roc_df)
    roc_df = get_cluster_roc(cluster_list, flag_list) # some redundant calculation here...
    auc = metrics.auc(roc_df['fpr'], roc_df['tpr'])
    if ( abs(ref_auc - auc) > 1e-6 ):  # about the same, instead of ref_auc != auc
        print(f"AUC difference! auc={auc}, ref_auc={ref_auc}")
    return { 
        'num_clusters':len(set(cluster_list)), 
        'mean_entropy': np.mean(cluster_stats['entropy']), 'mean_purity': np.mean(cluster_stats['purity']),
        'delta_entropy': cluster_improvement_H,
        'roc': roc_df, 'auc': auc, 'ref_auc': ref_auc
    }


def get_cluster_performance_df(clusters_df, flags_df, flag_category_map=None):
    """
    Compute a dataframe of cluster performance statistics (calls 'get_performance_for_clustering' for each cluster/flag pair).
    flags_df: dataframe where each column contains binary labels and rows correspond with cluster_assignments.
    clusters_df: dataframe where each column contains cluster assignments.
    returns: A dataframe containing cluster performance stats as returned by 'get_roc_for_clustering'.
    """
    import pandas as pd
    
    cluster_performance_rows = []
    
    for flag_col in flags_df.columns:
        flags = flags_df[flag_col]
        for cluster_col in clusters_df.columns:
            clusters = clusters_df[cluster_col]
            perf = get_performance_for_clustering(clusters, flags)
            perf['flag'] = flag_col
            perf['cluster_col'] = cluster_col
            cluster_performance_rows.append(perf)

    cpdf = pd.DataFrame(cluster_performance_rows)
    if flag_category_map is not None:
        cpdf['flag_category'] = [flag_category_map[f] for f in cpdf['flag']]
    
    return cpdf

def plot_auc_vs_delta_entropy(my_cluster_performance, category='framework', file=''):
    cpdf = my_cluster_performance.copy().reset_index(drop=True)
    cpdf['log_delta_entropy']  = np.log(cpdf.delta_entropy)
    
    if 'flag_category' in cpdf.columns:
        cpdf = cpdf.loc[cpdf.flag_category == category]
        ax = sns.lineplot(x='auc', y='log_delta_entropy', data=cpdf, hue='flag', style='flag_category')
    else:
        ax = sns.lineplot(x='auc', y='log_delta_entropy', data=cpdf, hue='flag')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    if file != '':
        plt.savefig(file)
        
    plt.show()

def plot_cluster_performance(my_cluster_performance, metric='auc', title='', file=''):
    """
    Plot cluster performance metric vs. cluster size, from the data returned by 'extract_cluster_aucs'

    my_cluster_performance: a pandas dataframe as returned by 'get_cluster_performance_df'. 
        If this dataframe has a 'flag_category' column, it will determine the line style.
    metric: one of 'auc' (default), 'mean_purity', 'mean_entropy'
    title: the title for the plot.
    file: Name of file where plot should be saved (leave as an empty string for no plot)
    usage:
        my_cluster_performance = get_cluster_performance_df(cluster_assignment_df, pattern_flags)
        plot_cluster_performance(my_cluster_performance, metric='auc', title='My awesome results')

    """
    # Plot 'num_cluster' values as ordinal categories.
    # Because we will be messing with data types, make sure we are only operating on a copy of the dataframe.
    cpdf = my_cluster_performance.copy().reset_index(drop=True)

    from pandas.api.types import is_numeric_dtype
    if not is_numeric_dtype(cpdf['num_clusters']):
        raise Exception("num_clusters must be numeric")
    
    # sort levels as numbers before converting to strings
    num_cluster_levels = [str(x) for x in sorted(set(cpdf['num_clusters']))]
    
    num_cluster_str = [str(x) for x in cpdf['num_clusters']]
    
    cpdf['num_clusters'] = pd.Series(
        pd.Categorical(
            num_cluster_str, categories=num_cluster_levels, ordered=True
        )
    )

    if 'flag_category' in cpdf.columns:
        ax = sns.lineplot(x='num_clusters', y=metric, data=cpdf, hue='flag', style='flag_category')
    else:
        ax = sns.lineplot(x='num_clusters', y=metric, data=cpdf, hue='flag')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.title(title)
    if metric == 'auc':
        plt.ylim(0.5, 1.05)
        
    if file != '':
        plt.savefig(file)
        
    plt.show()


def get_cluster_assignments(my_dendrogram, num_slices=8):
    """
    Produce a table of flattened cluster assignments from a dendrogram, where 
    each column represents a slice through the dendrogram producing a specific number of clusters.
    
    dendrogram: a hierarchical clustering structure as produced by a scipy.cluster.hierarchy linkage function like 'ward'.
    returns: a dataframe
    """
    import pandas as pd
    from scipy.cluster.hierarchy import fcluster
    
    cluster_cols = {}
    LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(num_slices):
        num_clusters = 2**(i+1)
        col_name = LETTERS[i]
        cluster_labels = fcluster(my_dendrogram, num_clusters, criterion='maxclust')
        cluster_cols[col_name] = cluster_labels
    
    return pd.DataFrame.from_dict(cluster_cols)


def hue_spectrum(num_colors=8, hue=1/3, saturation=0.5):
    """
    Generate a list of hex color values of a given hue and saturation with varying values of lightness.
    """
    import colorsys
    rgb_list = [ colorsys.hls_to_rgb(hue,lightness_lvl/(num_colors+1),saturation) for lightness_lvl in range(1,num_colors + 1)]
    hex_list = [f"#{round(r*255):02X}{round(g*255):02X}{round(b*255):02X}" for (r,g,b) in rgb_list]
    return reversed(hex_list)


def plot_aspect_roc_curves(aspect, cluster_performance_df): # wascluster_roc_df):
    """
    Plot the set of ROC curves (over different numbers of clusters) for a given aspect.
    aspect: the name of the binary label
    cluster_performance_df: dataframe returned by 'get_cluster_performance'
    """
    # To Do
    # 1. Re-do ROC calculation from scratch so we can account for each 
    #    individual cluster with its own line segment.
    #    The current implementation gives each unique score its own line segment, 
    #    so it is prone to combine all the clusters with frequencies of exactly 1 or exactly 0.
    # 2a. Support multiple distinguishable plotting style specifications, so we can
    #     compare the curves for differently weighted embeddings on the same plot.
    # 2b. Include a color spectrum for each plotting style, e.g, shades of red for one weighting, 
    #     and shades of blue for the other; `colorsys.hls_to_rgb` is our friend.

    import matplotlib.pyplot as plt

    aspect_roc_df = cluster_performance_df[ cluster_performance_df['flag'] == aspect ]

    plotspec = {'linewidth': 2, 'markersize': 8}
    plotspec1 = { #'color':'green', 
                 'marker': 'x', 'linestyle': 'solid', **plotspec}
    plotspec2 = { #'color':'red',   
                 'marker': '+', 'linestyle': 'dotted', **plotspec}
    
    num_rocs = len(aspect_roc_df)
    aucs = []
    for row in aspect_roc_df.to_dict(orient='records'):
        plt.plot( row['roc']['fpr'], row['roc']['tpr'], **plotspec1)
        aucs.append(row['auc'])
    plt.title(aspect)
    plt.show()
    print(num_rocs, [f"{auc:0.3f}" for auc in aucs])


def crosstab_clustermap(df, cluster_col_1, cluster_col_2):
    """
    Plot a clustermap of the cross-tabulation between two different ways of clustering the same items.
    
    df: a dataframe containning cluster assignment columns
    cluster_col_1, cluster_col_2: Names of two cluster columns in the dataframe.
    """
    import pandas as pd
    import seaborn as sns
    Xtab = pd.crosstab(df[cluster_col_1], df[cluster_col_2])
    sns.clustermap(Xtab)


