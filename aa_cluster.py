### Aspect aware clustering functions ###

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


def get_roc_for_clustering(cluster_list, flag_list):
    """
    Describe how well clusters predict binary labels (flags) by using frequency of positive labels in 
    clusters as a score to predict the label, and computing AUC.

    df:
    cluster_col:
    flag_col: 
    """
    from sklearn import metrics
    df = pd.DataFrame.from_dict({'cluster' : cluster_list, 'flag': flag_list})
    cluster_frequency = df.groupby('cluster').mean()
    scores = [ cluster_frequency['flag'][cluster_name] for cluster_name in df['cluster'] ]
    fpr, tpr, thresholds = metrics.roc_curve(df['flag'], scores)
    auc = metrics.auc(fpr, tpr)
    return { 'roc': {'fpr': fpr, 'tpr': tpr, 'threshold': thresholds}, 'auc': auc, 'num_clusters':len(set(cluster_list)) }


def get_cluster_rocs(flags_df, clusters_df):
    """
    Compute ROC curves based on cluster assignments.
    flags_df: dataframe where each column contains binary labels and rows correspond with cluster_assignments.
    clusters_df: dataframe where each column contains cluster assignments.
    returns: A dict of dicts where the keys are flag, then cluster, and the values are dicts as returned by 'get_roc_for_clustering'.
        An ROC dict has two keys, 'roc' (where the values are lists for tpr, fpr, and thresholds), and 'auc'.
    """
    import pandas as pd
    
    flag_cluster_roc = {}
    
    for flag_col in flags_df.columns:
        flags = flags_df[flag_col]
        for cluster_col in clusters_df.columns:
            clusters = clusters_df[cluster_col]
            if flag_col not in flag_cluster_roc:
                flag_cluster_roc[flag_col] = {}
            flag_cluster_roc[flag_col][cluster_col] = get_roc_for_clustering(clusters, flags)

    return flag_cluster_roc


def extract_cluster_roc_df(my_flag_cluster_roc):
    # To do: combine this function with `get_cluster_rocs` so we dont have to get them then extract them. Just put them in a dataframe to start with
    return pd.DataFrame([
        {
            'flag': flag_col,
            'num_clusters': my_flag_cluster_roc[flag_col][cluster_col]['num_clusters'],
            'auc': my_flag_cluster_roc[flag_col][cluster_col]['auc'],
            'roc': my_flag_cluster_roc[flag_col][cluster_col]['roc'],
        }
        for flag_col in my_flag_cluster_roc
            for cluster_col in my_flag_cluster_roc[flag_col]
    ])


def plot_cluster_aucs(my_cluster_aucs):
    """
    Plot cluster AUC vs. cluster size, from the data returned by 'extract_cluster_aucs'

    usage:
        cluster_rocs = get_cluster_rocs(pattern_flags, cluster_assignment_df)
        cluster_aucs =  extract_cluster_aucs(cluster_rocs)
        plot_cluster_aucs(cluster_aucs)

    """
    sns.lineplot(x='num_clusters', y='auc', data=my_cluster_aucs, hue='flag')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)


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


def plot_aspect_roc_curves(aspect, cluster_roc_df):
    # To Do
    # 1. Re-do ROC calculation from scratch so we can account for each 
    #    individual cluster with its own line segment.
    #    The current implementation gives each unique score its own line segment, 
    #    so it is prone to combine all the clusters with frequencies of exactly 1 or exactly 0.
    # 2. Support multiple distinguishable plotting style specifications, so we can
    #    compare the curves for differently weighted embeddings on the same plot.
    # 3. Include a color spectrum for each plotting style, e.g, shades of red for one weighting, 
    #    and shades of blue for the other; `colorsys.hls_to_rgb` is our friend.

    import matplotlib.pyplot as plt

    aspect_roc_df = cluster_roc_df[ cluster_roc_df['flag'] == aspect ]

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
    print(num_rocs, [f"{auc:0.3f}" for auc in aucs])
    