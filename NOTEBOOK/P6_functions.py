# Printing total nb and percentage of null

import pandas as pd

def print_null_pct(df):
    tot_null = df.isna().sum().sum()
    print('nb of null: ', tot_null, '\npct of null: ',
        '{:.1f}'.format(tot_null*100/(df.shape[0]*df.shape[1])))
    

'''Computes the projection of the observations of df on the two first axes of
a transformation (PCA, UMAP or t-SNE)
The center option (clustering model needed) allows to project the centers
on the two axis for further display, and to return the fitted model
NB: if the model wa already fitted, does not refit.'''

from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE


'''Takes the H matrix (topics/words) as a dataframe, extracts the n top words
and plots a wordcloud of the (n_top_words) top words for each topic.
'''

from wordcloud import WordCloud

def plot_wordclouds_topwords(H, n_top_words, n_rows=1, figsize=(18,8),
                             random_state=None):

    fig = plt.figure(figsize=figsize)
    wc = WordCloud(stopwords=None, background_color="black",
                   colormap="Dark2", max_font_size=150,
                   random_state=random_state)
    # boucle sur les thèmes
    for i, topic_name in enumerate(H.index,1):
        ser_10w_topic = H.loc[topic_name]\
            .sort_values(ascending=False)[0:n_top_words]
        wc.generate(' '.join(list(ser_10w_topic.index)))
        n_tot = H.index.shape[0]
        n_cols = (n_tot//n_rows)+((n_tot%n_rows)>0)*1
        ax = fig.add_subplot(n_rows,n_cols,i)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.title(topic_name, fontweight='bold')
        
    plt.show()

'''Computes the projection of the observations of df on the two first axes of
a transformation (PCA, UMAP or t-SNE)
The center option (clustering model needed) allows to project the centers
on the two axis for further display, and to return the fitted model
NB: if the model wa already fitted, does not refit.'''

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE

def prepare_2D_axes(df, ser_clust=None, proj=['PCA', 'UMAP', 't-SNE'],
                    model=None, centers_on=False, random_state=14):

    dict_proj = dict()

    if centers_on:  # Compute and include the centers in the points
        if model is not None:
            model = model.fit(df) if not is_fitted(model) else model
            # ### all clusterers don't have .cluster_centers method -> changed
            # centers = model.cluster_centers_ 
            # ind_centers = ["clust_" + str(i) for i in range(centers.shape[0])]
            # centers_df = pd.DataFrame(centers,
            #                           index=ind_centers,
            #                           columns=df.columns)
            #### all clusterers don't have .predict/labels_ method -> changed
            if hasattr(model, 'labels_'):
                clust = model.labels_
            else:
                clust = model.predict(df)
        else:
            clust = ser_clust
        # calculation of centers
        centers_df = df.assign(clust=clust).groupby('clust').mean()
        df = df.append(centers_df)

    ## Projection of all the points through the transformations

    # PCA
    if 'PCA' in proj:
        pca = PCA(n_components=2, random_state=random_state)
        df_proj_PCA_2D = pd.DataFrame(pca.fit_transform(df),
                                      index=df.index,
                                      columns=['PC' + str(i) for i in range(2)])
        dict_proj = dict({'PCA': df_proj_PCA_2D})

    # UMAP
    if 'UMAP' in proj:
        umap = UMAP(n_components=2, random_state=random_state)
        df_proj_UMAP_2D = pd.DataFrame(umap.fit_transform(df),
                                       index=df.index,
                                       columns=['UMAP' + str(i) for i in range(2)])
        dict_proj = dict({'UMAP': df_proj_UMAP_2D})

    # t-SNE
    if 't-SNE' in proj:
        tsne = TSNE(n_components=2, random_state=random_state)
        df_proj_tSNE_2D = pd.DataFrame(tsne.fit_transform(df),
                                       index=df.index,
                                       columns=['t-SNE' + str(i) for i in range(2)])
        dict_proj = dict({'t-SNE': df_proj_tSNE_2D})

    # Separate the clusters centers from the other points if center option in on
    if centers_on:
        dict_proj_centers = {}
        for name, df_proj in dict_proj.items():
            dict_proj_centers[name] = dict_proj[name].loc[centers_df.index]
            dict_proj[name] = dict_proj[name].drop(index=centers_df.index)
        return dict_proj, dict_proj_centers, model
    else:
        return dict_proj


''' Plots the points on two axis (projection choice available : PCA, UMAP, t-SNE)
with clusters coloring if model available (grey if no model given).
NB: if the model wa already fitted, does not refit.'''

import seaborn as sns

def plot_projection(df, model=None, ser_clust = None, proj='PCA', title=None,
                    figsize=(5, 3), size=1, palette='tab10',
                    legend_on=False, fig=None, ax=None, random_state=14):

    fig = plt.figure(figsize=figsize) if fig is None else fig
    ax = fig.add_subplot(111) if ax is None else ax

    # a1 - if model : computes clusters, clusters centers and plot with colors
    if model is not None:

        # Computes the axes for projection with centers
        # (uses fitted model if already fitted)
        dict_proj, dict_proj_centers, model = prepare_2D_axes(df,
                                                              proj=[proj],
                                                              model=model,
                                                              centers_on=True,
                                                              random_state=random_state)

        # ...or using model already fitted in prepare_2D_axes to get it
        #### all clusterers don't have .predict/labels_ method -> changed
        if hasattr(model, 'labels_'):
            clust = model.labels_
        else:
            clust = model.predict(df)
        ser_clust = pd.Series(clust,
                                index=df.index,
                                name='Clust')
        
    # a2 - if no model but ser_clust is given, plot with colors
    elif ser_clust is not None:
        
        # Computes the axes for projection
        dict_proj, dict_proj_centers, _ = \
            prepare_2D_axes(df, ser_clust=ser_clust, proj=[proj],
                            model=None, centers_on=True,
                            random_state=random_state)

        n_clust = ser_clust.nunique()
        colors = sns.color_palette(palette, n_clust).as_hex()

    # b1 - if ser_clust exists (either calculated from model or given)
    if ser_clust is not None:

        # Showing the points, cluster by cluster
        # for i in range(n_clust):
        for i, name_clust in enumerate(ser_clust.unique()):
            ind = ser_clust[ser_clust == name_clust].index
            ax.scatter(dict_proj[proj].loc[ind].iloc[:, 0],
                       dict_proj[proj].loc[ind].iloc[:, 1],
                       s=size, alpha=0.7, c=colors[i], zorder=1)

            # Showing the clusters centers
            ax.scatter(dict_proj_centers[proj].iloc[:, 0].loc[name_clust],
                        dict_proj_centers[proj].iloc[:, 1].loc[name_clust],#.values[i],
                        marker='o', c=colors[i], alpha=0.7, s=150,
                       edgecolor='k',
                       label=f"{i}: {name_clust}", zorder=10)
            # Showing the clusters centers labels (number)
            ax.scatter(dict_proj_centers[proj].iloc[:, 0].loc[name_clust],#.values[i],
                        dict_proj_centers[proj].iloc[:, 1].loc[name_clust],
                        marker=r"$ {} $".format(i),#
                        c='k', alpha=1, s=70, zorder=100)
            if legend_on:
                plt.legend()
                ax.legend().get_frame().set_alpha(0.3)


    # b2 - if no ser_clust: only plot points in grey
    else:
        # Computes the axes for projection without centers
        dict_proj = prepare_2D_axes(df,
                                    proj=[proj],
                                    centers_on=False,
                                    random_state=random_state)
        # Plotting the point in grey
        ax.scatter(dict_proj[proj].iloc[:, 0],
                   dict_proj[proj].iloc[:, 1],
                   s=size, alpha=0.7, c='grey')

    title = "Projection: " + proj if title is None else title
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('ax 1'), ax.set_ylabel('ax 2')



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

''' Takes a pd.Series containing the texts of each description
applies a preprocessing function if given (stopwords, stemming...)
then turn the descriptions in vectors (bow of tf-idf, depending on the avlue of
 tfidf_on)
 returns document term matrix as a dataframe and the list of new excluded words.
'''

def compute_doc_terms_df(ser_desc, 
                         preproc_func=None,
                         preproc_func_params=None,
                         vec_params = {'min_df': 1},
                         tfidf_on=False,
                         print_opt=False):

    # ---- Apply a preprocessing function prior to vectorization
    if preproc_func is not None:
        ser_desc = ser_desc.apply(lambda x: preproc_func(x,
                                                         **preproc_func_params))
        ser_desc = ser_desc.apply(lambda x: ' '.join(x))
    else:
        ser_desc = ser_desc
    
    # ---- Vectorization of each of the texts (row)
    if tfidf_on:
        # TF-IDF matrix
        vec = TfidfVectorizer(**vec_params)
    else:
        # BOW matrix (count)
        vec = CountVectorizer(**vec_params)

    doc_term = vec.fit_transform(ser_desc)
    if print_opt:
        print( "Created %d X %d doc_term matrix" % (doc_term.shape[0],
                                                    doc_term.shape[1]))

    # ---- Vocabulary of the document_term matrix
    doc_term_voc = vec.get_feature_names()
    if print_opt:
        print("Vocabulary has %d distinct terms" % len(doc_term_voc))

    # ---- Get the list of the new stop-words
    new_sw = vec.stop_words_
    if print_opt:
        print("Old stop-words list has %d entries" % len(sw) )
        print("New stop-words list has %d entries" % len(new_sw))

    doc_term_df = pd.DataFrame(doc_term.todense(),
                index=ser_desc.index, # each item
                columns=doc_term_voc) # each word

    # document term matrix as a dataframe and the list of new excluded words
    return doc_term_df, new_sw



'''
Takes a vectorized matrix (dataframe) of the documents
(Document-trem matrix: BOW or tf-idf... documents(rows) x words (columns))
and returns the projected vectors in the form of a dataframe
(words (rows) x w2v dimensions(columns))
'''

def proj_term_doc_on_w2v(term_doc_df, w2v_model, print_opt=False):

    # Checking the number of words of our corpus existing in the wiki2vec dictionary
    li_common_words = []
    for word in term_doc_df.columns:
        word_ = w2v_model.get_word(word)
        if word_ is not None:
            li_common_words.append(word)
    if print_opt:
        print(f"The w2v dictionary contains {len(li_common_words)} words out of \
the {term_doc_df.shape[1]} existing in our descriptions,\ni.e. \
{round(100*len(li_common_words)/term_doc_df.shape[1],1)}% of the whole vocabulary.")

    # extracting each of the word vectors
    word_vectors_df = pd.DataFrame()
    for word in li_common_words:
        word_vectors_df[word] = w2v_model.get_word_vector(word)
    word_vectors_df = word_vectors_df.T
    word_vectors_df.columns = ['dim_'+str(i)\
                               for i in range(word_vectors_df.shape[1])]

    # projection of the Document_terms matrix on the wiki2vec
    w2v_emb_df = term_doc_df[li_common_words].dot(word_vectors_df)

    return w2v_emb_df


''' Takes an image, resizes the image and fills the non existing space
with white 
'''

import cv2
import numpy as np
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur

def preproc_image(img, size=224, fill_col=(255,255,255),
                  autocontrast = False, equalize=False,
                  gauss_blur = None, interpolation=cv2.INTER_AREA):

    img = Image.fromarray(img)
    if autocontrast:
        img = ImageOps.autocontrast(img)
    if equalize:
        img = ImageOps.equalize(img)
    if gauss_blur is not None:
        img = img.filter(GaussianBlur(radius=gauss_blur))

    w, h = img.size
    if h == w:
        new_img = img
    else:
        dif = h if h > w else w
        new_img = Image.new('RGB', (dif, dif), fill_col)
        new_img.paste(img, (int((dif - w) / 2), int((dif - h) / 2)))

    return cv2.resize(np.asarray(new_img), (size, size), interpolation)



""" For a each number of clusters in a list ('list_n_clust'),
- runs iterations ('n_iter' times) of a KMeans on a given dataframe,
- computes the 4 scores : silhouette, davies-bouldin, calinski_harabasz and
distortion
- if enabled only('return_pop'): the proportion (pct) of the clusters
for each iteration and number of clusters
- and returns 3 dictionnaries:
    - dict_scores_iter: the 4 scores
    - dict_ser_clust_n_clust: the list of clusters labels for df rows
    - if enabled only (return_pop), 'dict_pop_perc_n_clust' : the proportions

NB: the functions 'plot_scores_vs_n_clust', 'plot_prop_clust_vs_nclust' and
'plot_clust_prop_pie_vs_nclust' plot
respectively:
- the scores vs the number of clusters,
- the proportion of clusters
- and the pies of the clusters ratio,
 based on the dictionnaries provided by compute_clust_scores_nclust"""


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier

def compute_clust_scores_nclust(df, df_fit=None, model=None, n_iter=10, inertia=True,
                                list_n_clust=range(2,8), return_pop=False):
#### ATTENTION AU CAS PAR CAS POUR LES MODELES AUTRES QUE KMEANS

    dict_pop_perc_n_clust = {}
    dict_ser_clust_n_clust = {}
    dict_scores_iter = {}

    df_fit = df if df_fit is None else df_fit
    km_default = True if model is None else False
    ahc = True if model == 'ahc' else False
    gmm = True if model == 'gmm' else False

    # --- Looping on the number of clusters to compute the scores
    for i, n_clust in enumerate(list_n_clust,1):

        silh, dav_bould, cal_harab, distor = [], [], [], []
        pop_perc_iter, ser_clust_iter = pd.DataFrame(), pd.DataFrame()

        # Iterations of the same model (stability)
        for j in range(n_iter):
            if km_default:
                model = KMeans(n_clusters=n_clust,
                               random_state=np.random.randint(100))
            elif ahc:
                ahc = AgglomerativeClustering(n_clusters=n_clust)
                clf = RandomForestClassifier(random_state=np.random.randint(100))
                model = InductiveClusterer(ahc, clf)
            elif gmm:
                # reinitialisation of the random state (gmm)
                model = GaussianMixture(n_components=n_clust,
                                        covariance_type='spherical',
                                        random_state=np.random.randint(100))
            else:
                print("ERROR: unknown model asked")
            model.fit(df_fit)
            ser_clust = pd.Series(data=model.predict(df),
                                  index=df.index,
                                  name="iter_"+str(j))
            ser_clust_iter = pd.concat([ser_clust_iter, ser_clust.to_frame()],
                                       axis=1)

            if return_pop:
                # Compute pct of clients in each cluster
                pop_perc = 100 * ser_clust.value_counts() / df.shape[0]
                pop_perc.sort_index(inplace=True)
                pop_perc.index = ['clust_'+str(i) for i in pop_perc.index]
                pop_perc_iter = pd.concat([pop_perc_iter, pop_perc.to_frame()],
                                          axis=1)
        
            # Computing scores for iterations
            silh.append(silhouette_score(X=df, labels=ser_clust))
            dav_bould.append(davies_bouldin_score(X=df, labels=ser_clust))
            cal_harab.append(calinski_harabasz_score(X=df, labels=ser_clust))
            if inertia: distor.append(model.inertia_)

        dict_ser_clust_n_clust[n_clust] = ser_clust_iter

        if return_pop:
            # dict of the population (pct) of clusters iterations
             dict_pop_perc_n_clust[n_clust] = pop_perc_iter.T

        # Dataframe of the results on iterations
        scores_iter = pd.DataFrame({'Silhouette': silh,
                                    'Davies_Bouldin': dav_bould,
                                    'Calinsky_Harabasz': cal_harab,
                                    })
        if inertia:
            scores_iter['Distortion'] = distor
        dict_scores_iter[n_clust] = scores_iter

    if return_pop:
        return dict_scores_iter, dict_ser_clust_n_clust, dict_pop_perc_n_clust
    else:
        return dict_scores_iter, dict_ser_clust_n_clust


''' Plot the 4 mean scores stored in the dictionnary returned by the function
compute_clust_scores_nclust (dictionnary of dataframes of scores (columns)
for each iteration (rows) of the model and for each number of clusters
in a figure with error bars (2 sigmas)'''

def plot_scores_vs_n_clust(dict_scores_iter, figsize=(15,3)):

    fig = plt.figure(figsize=figsize)
    list_n_clust = list(dict_scores_iter.keys())

    # Generic fonction to unpack dictionary and plot one graph
    def score_plot_vs_nb_clust(scores_iter, name, ax, c=None):
        score_mean = [dict_scores_iter[i].mean().loc[n_score] for i in list_n_clust]
        score_std = np.array([dict_scores_iter[i].std().loc[n_score]\
                            for i in list_n_clust])
        ax.errorbar(list_n_clust, score_mean, yerr=2*score_std, elinewidth=1,
                capsize=2, capthick=1, ecolor='k', fmt='-o', c=c, ms=5,
                barsabove=False, uplims=False)

    li_scores = ['Silhouette', 'Davies_Bouldin',
                 'Calinsky_Harabasz', 'Distortion']
    li_colors = ['r', 'b', 'purple', 'g']

    # Looping on the 4 scores
    i=0
    for n_score, c in zip(li_scores, li_colors):
        i+=1
        ax = fig.add_subplot(1,4,i)
        
        score_plot_vs_nb_clust(dict_scores_iter, n_score, ax, c=c)
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel(n_score+' score')

    fig.suptitle('Clustering score vs. number of clusters',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()


''' Plot the proportion (%) of each cluster (columns) returned by the function
compute_clust_scores_nclust (dictionnary of dataframes of the proportion
for each iteration (rows) of the model in one figure with error bars (2 sigmas)'''

def plot_prop_clust_vs_nclust(dict_pop_perc_n_clust, figsize=(15,3)):

    fig = plt.figure(figsize=figsize)
    list_n_clust = list(dict_pop_perc_n_clust.keys())
    

    for i, n_clust in enumerate(list_n_clust, 1):
        n_iter = dict_pop_perc_n_clust[n_clust].shape[0]
        ax = fig.add_subplot(3,3,i)
        sns.stripplot(data=dict_pop_perc_n_clust[n_clust],
                      edgecolor='k', linewidth=1,  ax=ax)
        ax.set(ylim=(0,100))
        ax.set_ylabel("prop. of the clusters (%)")
    fig.suptitle(f"Proportion of the clusters through {n_iter} iterations",
                fontweight='bold', fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.97])


""" Plot pies of the proportion of the clusters using the proportions
stored in the dictionnary returned by the function
'compute_clust_scores_nclust' (dictionnary of dataframes of the
proportions (columns) for each iteration (rows) of the model
and for each number of clusters in a figure with error (+/-2 sigmas)"""

def plot_clust_prop_pie_vs_nclust(dict_pop_perc_n_clust,
                                  list_n_clust, figsize=(15, 3)):

    fig = plt.figure(figsize=figsize)

    for i, n_clust in enumerate(list_n_clust,1):
        ax = fig.add_subplot(str(1) + str(len(list_n_clust)) + str(i))

        mean_ = dict_pop_perc_n_clust[n_clust].mean()
        std_ = dict_pop_perc_n_clust[n_clust].std()
        
        wedges, texts, autotexts = ax.pie(mean_,
                autopct='%1.0f%%',
                labels=["(+/-{:.0f})".format(i) for i in std_.values],
                pctdistance=0.5)
        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=8)
        ax.set_title(f'{str(n_clust)} clusters')  # , pad=20

    fig.suptitle('Clusters ratio', fontsize=16, fontweight='bold')
    plt.show()


''' Plots on the left the silhouette scores of each cluster and
on the right the projection of the points with cluster labels as cluster'''

from sklearn.metrics import silhouette_score, silhouette_samples

def silh_scores_vs_n_clust(df, n_clust, proj='PCA',
                           xlim=(-0.1, 0.8), figsize=(18, 3), palette='tab10'):
    
    palette = sns.color_palette(palette, np.max(n_clust))
    colors = palette.as_hex()

    distor = []
    for n in n_clust:
        fig = plt.figure(1, figsize=figsize)

        # --- Plot 1: Silhouette scores
        ax1 = fig.add_subplot(121)

        model = KMeans(n_clusters=n, random_state=14)
        model = model.fit(df)

        ser_clust = pd.Series(model.predict(df),
                              index=df.index,
                              name='Clust')
        distor.append(model.inertia_)
        sample_silh_val = silhouette_samples(df, ser_clust)

        y_lower = 10
        # colors = [colors[x] for x in ser_clust.astype('int')]
        for i in range(n):
            # Aggregate and sort silh scores for samples of clust i
            clust_silh_val = sample_silh_val[ser_clust == i]
            clust_silh_val.sort()
            size_clust = clust_silh_val.shape[0]
            y_upper = y_lower + size_clust
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0,
                              clust_silh_val,
                              facecolor=colors[i],
                              edgecolor=colors[i],
                              alpha=0.7)

            # Label of silhouette plots with their clust. nb. at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_clust, str(i))

            # Computes the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        silhouette_avg = silhouette_score(df, ser_clust)
        ax1.set_title("Nb of clusters: {} | Avg silhouette: {:.3f}" \
                      .format(n, silhouette_avg), fontsize=12)
        ax1.set_xlabel("Silhouette coeff. values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax1.set_xlim(list(xlim))
        # (n+1)*10: inserting blank spaces between clusters silh scores
        ax1.set_ylim([0, df.shape[0] + (n + 1) * 10])

        # --- Plot 2: Showing clusters on chosen projection
        ax2 = fig.add_subplot(122)
        # uses already fitted model
        plot_projection(df, model=model,
                        proj=proj,
                        palette=palette,
                        fig=fig, ax=ax2)

        ax2.set_title('projection: ' + proj, fontsize=12)

        plt.suptitle("Silhouette analysis for {} clusters".format(n),
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


''' Generates the list of all unique combination of k numbers 
(no matter the order) among a given seq list of objects'''

def combinlist(seq, k):
    p = []
    i, imax = 0, 2**len(seq)-1
    while i<=imax:
        s = []
        j, jmax = 0, len(seq)-1
        while j<=jmax:
            if (i>>j)&1==1:
                s.append(seq[j])
            j += 1
        if len(s)==k:
            p.append(s)
        i += 1
    return p


'''Takes a dataframe of clusters number (prediction) for a set of observation, 
and computes the ARI score between pairs of columns.
Two modes are available:
- first_vs_others=False: to check the initialisation stability.
The columns are obtains for n_columns iterations of the same model
with different initialisation
- first_vs_others=True: to compare the predictions obtained with the whole
dataset (first column) and predictions obtained with a sample
(the other columns)
Return a pd.Series of the ARI scores (values) for each pair of columns (index).
'''

from sklearn.metrics import adjusted_rand_score

def ARI_column_pairs(df_mult_ser_clust, first_vs_others=False, print_opt=True):

    n_columns = len(df_mult_ser_clust.columns)
    n_clust = df_mult_ser_clust.stack().nunique()
    
    # Computes ARI scores for each pair of models
    ARI_scores = []
    if first_vs_others: # first columns versus the others
        pairs_list = [[df_mult_ser_clust.columns[0],
                       df_mult_ser_clust.columns[i]] \
                      for i in range(1, n_columns)]
        if print_opt: print("--- ARI between first and the {} others ---"\
                            .format(n_columns-1))
        name = f'ARI_{str(n_clust)}_clust_first_vs_others'
    else: # all pairs
        pairs_list = combinlist(df_mult_ser_clust.columns,2)
        if print_opt: print("--- ARI all {} unique pairs ---"\
                            .format(len(pairs_list)))
        name = f'ARI_{str(n_clust)}_clust_all_pairs'

    for i, j in pairs_list:
        ARI_scores.append(adjusted_rand_score(df_mult_ser_clust.loc[:,i],
                                              df_mult_ser_clust.loc[:,j]))

    # Compute the mean and standard deviation of ARI scores
    ARI_mean, ARI_std = np.mean(ARI_scores), np.std(ARI_scores)
    ARI_min, ARI_max = np.min(ARI_scores), np.max(ARI_scores)
    if print_opt: print("ARI: mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f} "\
            .format(ARI_mean, ARI_std, ARI_min, ARI_max))

    return pd.Series(ARI_scores, index=pd.Index(pairs_list),
                     name=name)

