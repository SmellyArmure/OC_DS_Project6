# Printing total nb and percentage of null

import pandas as pd

def print_null_pct(df):
    tot_null = df.isna().sum().sum()
    print('nb of null: ', tot_null, '\npct of null: ',
        '{:.1f}'.format(tot_null*100/(df.shape[0]*df.shape[1])))

# Plotting heatmap (2 options available, rectangle or triangle )

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_heatmap(corr, title, figsize=(8, 4), vmin=-1, vmax=1, center=0,
                 palette=sns.color_palette("coolwarm", 20), shape='rect',
                 fmt='.2f', robust=False, fig=None, ax=None):
    fig = plt.figure(figsize=figsize) if fig is None else fig
    ax = fig.add_subplot(111) if ax is None else ax

    if shape == 'rect':
        mask = None
    elif shape == 'tri':
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
    else:
        print('ERROR : this type of heatmap does not exist')

    palette = palette
    ax = sns.heatmap(corr, mask=mask, cmap=palette, vmin=vmin, vmax=vmax,
                     center=center, annot=True, annot_kws={"size": 10}, fmt=fmt,
                     square=False, linewidths=.5, linecolor='white',
                     cbar_kws={"shrink": .9, 'label': None}, robust=robust,
                     xticklabels=corr.columns, yticklabels=corr.index,
                     ax=ax)
    ax.tick_params(labelsize=10, top=False, bottom=True,
                   labeltop=False, labelbottom=True)
    ax.collections[0].colorbar.ax.tick_params(labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    ax.set_title(title, fontweight='bold', fontsize=12)



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



''' Takes a pd.Series containing the texts of each description
applies a preprocessing function if given (stopwords, stemming...)
then turn the descriptions in vectors (bow of tf-idf, depending on the avlue of
 tfidf_on)
 returns document term matrix as a dataframe and the list of new excluded words.
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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


''' Builds a customizable column_transformer which parameters can be optimized in a GridSearchCV
CATEGORICAL : three differents startegies for 3 different types of
categorical variables:
- low cardinality: customizable strategy (strat_low_card)
- high cardinality: customizable strategy (strat_high_card)
- boolean or equivalent (2 categories): ordinal
QUANTITATIVE (remainder): 
- StandardScaler

-> EXAMPLE (to use apart from gscv):
cust_enc = CustTransformer(thresh_card=12,
                       strat_binary = 'ord',
                       strat_low_card = 'ohe',
                       strat_high_card = 'loo',
                       strat_quant = 'stand')
cust_enc.fit(X_tr, y1_tr)
cust_enc.transform(X_tr).shape, X_tr.shape

-> EXAMPLE (to fetch names of the modified dataframe):
small_df = df[['Outlier', 'Neighborhood', 'CertifiedPreviousYear',
               'NumberofFloors','ExtsurfVolRatio']]
# small_df.head(2)
cust_trans = CustTransformer()
cust_trans.fit(small_df)
df_enc = cust_trans.transform(small_df)
cust_trans.get_feature_names(small_df)

'''
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce
from sklearn.preprocessing import *
import numpy as np
import pandas as pd


class CustTransformer(BaseEstimator):

    def __init__(self, thresh_card=12,
                 strat_binary='ord', strat_low_card='ohe',
                 strat_high_card='bin', strat_quant='stand'):
        self.thresh_card = thresh_card
        self.strat_binary = strat_binary
        self.strat_low_card = strat_low_card
        self.strat_high_card = strat_high_card
        self.strat_quant = strat_quant
        self.dict_enc_strat = {'binary': strat_binary,
                               'low_card': strat_low_card,
                               'high_card': strat_high_card,
                               'numeric': strat_quant}

    def d_type_col(self, X):
        bin_cols = (X.nunique()[X.nunique() <= 2].index)
        X_C_cols = X.select_dtypes(include=['object', 'category'])
        C_l_card_cols = \
            X_C_cols.nunique()[X_C_cols.nunique() \
                .between(3, self.thresh_card)].index
        C_h_card_cols = \
            X_C_cols.nunique()[X_C_cols.nunique() > self.thresh_card].index
        Q_cols = [c for c in X.select_dtypes(include=[np.number]).columns \
                  if c not in bin_cols]
        d_t = {'binary': bin_cols,
               'low_card': C_l_card_cols,
               'high_card': C_h_card_cols,
               'numeric': Q_cols}
        d_t = {k: v for k, v in d_t.items() if len(v)}
        # print(d_t)
        return d_t

    def get_feature_names(self, X, y=None):
        if self.has_num and self.has_cat:
            self.ct_cat.fit(X, y)
            cols = self.ct_cat.get_feature_names() + self.num_cols
        elif self.has_num and not self.has_cat:
            cols = self.num_cols
        elif not self.has_num and self.has_cat:
            self.ct_cat.fit(X, y)
            cols = self.ct_cat.get_feature_names()
        else:
            cols = None
        return cols

    def fit(self, X, y=None):
        # Dictionary to translate strategies
        d_enc = {'ohe': ce.OneHotEncoder(),
                 'hash': ce.HashingEncoder(),
                 'ord': ce.OrdinalEncoder(),
                 'loo': ce.LeaveOneOutEncoder(),
                 'bin': ce.BinaryEncoder(),
                 'stand': StandardScaler(),
                 'minmax': MinMaxScaler(),
                 'maxabs': MaxAbsScaler(),
                 'robust': RobustScaler(quantile_range=(25, 75)),
                 'norm': Normalizer(),
                 'quant_uni': QuantileTransformer(output_distribution='uniform'),
                 'quant_norm': QuantileTransformer(output_distribution='normal'),
                 'boxcox': PowerTransformer(method='box-cox'),
                 'yeo': PowerTransformer(method='yeo-johnson'),
                 'log': FunctionTransformer(func=lambda x: np.log1p(x),
                                            inverse_func=lambda x: np.expm1(x)),
                 'none': FunctionTransformer(func=lambda x: x,
                                             inverse_func=lambda x: x),
                 }

        # # dictionnaire liste des transfo categorielles EXISTANTES
        d_t = self.d_type_col(X)
        # numerics
        self.has_num = ('numeric' in d_t.keys())
        # categoricals
        self.has_cat = len([s for s in d_t.keys() if s in ['binary', 'low_card', 'high_card']]) > 0
        if self.has_cat:
            list_trans = []  # dictionnaire des transfo categorielles EXISTANTES
            for k, v in d_t.items():
                if k != 'numeric':
                    list_trans.append((k, d_enc[self.dict_enc_strat[k]], v))

            self.cat_cols = []  # liste des colonnes catégorielles à transformer
            for k, v in self.d_type_col(X).items():
                if k != 'numeric': self.cat_cols += (list(v))

            self.ct_cat = ColumnTransformer(list_trans)
            self.cat_trans = Pipeline([("categ", self.ct_cat)])

        if self.has_num:
            self.num_trans = Pipeline([("numeric", d_enc[self.strat_quant])])
            self.num_cols = d_t['numeric']

        if self.has_num and self.has_cat:
            self.column_trans = \
                ColumnTransformer([('cat', self.cat_trans, self.cat_cols),
                                   ('num', self.num_trans, self.num_cols)])
        elif self.has_num and not self.has_cat:
            self.column_trans = \
                ColumnTransformer([('num', self.num_trans, self.num_cols)])
        elif not self.has_num and self.has_cat:
            self.column_trans = ColumnTransformer([('cat', self.cat_trans, self.cat_cols)])
        else:
            print("The dataframe is empty : no transformation can be done")

        return self.column_trans.fit(X, y)

    # OLD VERSION WITHOUT NAME OF THE COLUMNS, OUTPUT AS A NP.ARRAY
    # def transform(self, X, y=None): 
    #     return  self.column_trans.transform(X)
    # def fit_transform(self, X, y=None):
    #     if y is None:
    #         self.fit(X)
    #         return self.column_trans.transform(X)
    #     else:
    #         self.fit(X, y)
    #         return self.column_trans.transform(X)

    def transform(self, X, y=None):  # to get a dataframe
        return pd.DataFrame(self.column_trans.transform(X),
                            index=X.index,
                            columns=self.get_feature_names(X, y))

    def fit_transform(self, X, y=None):
        if y is None:
            self.fit(X)
            return pd.DataFrame(self.column_trans.transform(X),
                                index=X.index,
                                columns=self.get_feature_names(X, y))
        else:
            self.fit(X, y)
            return pd.DataFrame(self.column_trans.transform(X, y),
                                index=X.index,
                                columns=self.get_feature_names(X, y))



'''
Takes two series giving for each row :
- the true label
- the cluster label
Then computes the matrix counting each pair of true category/ cluster label.
Then reorder the lines and columns in order to have maximum diagonal.
The best bijective correspondance between categories and clusters is obtained by
 list(zip(result.columns, result.index))
'''

from sklearn.preprocessing import FunctionTransformer
from scipy.optimize import linear_sum_assignment

def confusion_matrix_clust(true_cat, clust_lab, normalize=False,
                           margins_sums=False, margins_score=False):

    ### Count the number of articles in eact categ/clust pair
    cross_tab = pd.crosstab(true_cat, clust_lab,
                         normalize=normalize)

    ### Rearrange the lines and columns to maximize the diagonal values sum
    # Take the invert values in the matrix
    func = lambda x: 1/(x+0.0000001)
    inv_func = lambda x: (1/x) - 0.0000001
    funct_trans = FunctionTransformer(func, inv_func)
    inv_df = funct_trans.fit_transform(cross_tab)

    # Use hungarian algo to find ind and row order that minimizes inverse
    # of the diag vals -> max diag vals
    row_ind, col_ind = linear_sum_assignment(inv_df.values)
    inv_df = inv_df.loc[inv_df.index[row_ind],
                        inv_df.columns[col_ind]]

    # Take once again inverse to go back to original values
    cross_tab = funct_trans.inverse_transform(inv_df)
    result = cross_tab.copy(deep='True')

    if normalize == False:
        result = result.round(0).astype(int)

    if margins_sums:
        # assign the sums margins to the result dataframe
        marg_sum_vert = result[result.columns].sum(1)
        result = result.assign(cat_sum=marg_sum_vert)
        marg_sum_hor = result.loc[result.index].sum(0)
        result = result.append(pd.Series(marg_sum_hor,
                                         index=result.columns,
                                         name='clust_sum'))

    if margins_score:
        # Compute a correpondance score between clusters and categories
        li_cat_clust = list(zip(cross_tab.index,
                                cross_tab.columns))
        li_cat_corresp_score, li_clust_corresp_score = [], []
        for i, tup in enumerate(li_cat_clust):
            match = result.loc[tup]
            cat_corr_score = 100*match/cross_tab.sum(1).iloc[i]
            clust_corr_score = 100*match/cross_tab.sum(0).iloc[i]
            li_cat_corresp_score.append(cat_corr_score)
            li_clust_corresp_score.append(clust_corr_score)

        # assign the margins to the result dataframe
        if margins_sums:
            li_cat_corresp_score.append('-')
            li_clust_corresp_score.append('-')

        marg_vert = pd.Series(li_cat_corresp_score,
                              index=result.index,
                              name='cat_matching_score_pct')
        result = result.assign(cat_matching_score_pct=marg_vert) 

        marg_hor = pd.Series(li_clust_corresp_score + ['-'],
                             index=result.columns,
                             name='clust_matching_score_pct')
        result = result.append(marg_hor)

    result = result.fillna('-')

    return result

    '''
Computing the trustworthiness category by category
'''
from sklearn.manifold import trustworthiness

def groups_trustworthiness(df, df_proj, ser_clust):
    
    gb_clust = df.groupby(ser_clust)
    tw_clust, li_clust = [], []
    for n_clust, ind_sub_df in gb_clust.groups.items():
        li_clust.append(n_clust)
        tw_clust.append(trustworthiness(df.loc[ind_sub_df],
                                        df_proj.loc[ind_sub_df],
                                        n_neighbors=5, metric='euclidean'))
    ser = pd.Series(tw_clust,
                    index=li_clust,
                    name='tw')
    return ser

''' Plots the points on two axis (projection choice available : PCA, UMAP, t-SNE)
with clusters coloring if model available (grey if no model given).
NB: if the model wa already fitted, does not refit.'''

import seaborn as sns
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import LabelEncoder

def plot_projection_cat_clust(df, model=None, ser_clust = None, true_cat=None,
                              proj='PCA', title=None, figsize=(5, 3),
                              size=1, edgelinesize=25, centersize=150,
                              palette='tab10', legend_on=False,
                              bboxtoanchor=None, fig=None, ax=None,
                              random_state=14):

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
            
    # Computing the global trustworthiness
    trustw = trustworthiness(df, dict_proj[proj],
                             n_neighbors=5, metric='euclidean')
    # Computing the trustworthiness category by category
    ser_tw_cat = groups_trustworthiness(df, dict_proj[proj], true_cat)
    ser_tw_clust = groups_trustworthiness(df, dict_proj[proj], ser_clust)

    # b1 - if ser_clust exists (either calculated from model or given)
    if ser_clust is not None:
        # Prepare the correpondance between categories and clusters
        cm = confusion_matrix_clust(df_res_clust['categories'],
                                    df_res_clust['NMF_tfidf'])
        cat_list = cm.index # liste des catégories, dans l'ordre
        clust_list = cm.columns # liste des clusters, dans l'ordre

        # prepare color values
        n_clust = ser_clust.nunique()
        c1 = sns.color_palette(palette, cm.shape[0]).as_hex() # cat
        c2 = sns.color_palette(palette, cm.shape[1]).as_hex() # clust
        colors1 = true_cat.map(dict(zip(cat_list, c1))) # traduit une catégorie en couleur de c1
        colors2 = ser_clust.map(dict(zip(clust_list, c2))) # traduit un cluster en couleur de c2

        # # prepare markers values
        # lenc = LabelEncoder()
        # markers = pd.Series([f'${i}$' for i in lenc.fit_transform(true_cat)],
        #                     index=true_cat.index)
        # Plot the data points
        for i in range(len(ser_clust.index)):
            ax.scatter(dict_proj[proj].iloc[i, 0],
                        dict_proj[proj].iloc[i, 1],
                        color=colors1.iloc[i],
                        s=size, linewidths=edgelinesize,
                        alpha=1, #marker=markers[i],
                        ec=colors2.iloc[i])

        # calculation of centers
        cent_clust_df = dict_proj[proj].assign(clust=ser_clust)\
                        .groupby(ser_clust).mean()         
        cent_cat_df = dict_proj[proj].assign(cat=true_cat)\
                        .groupby(true_cat).mean()
        
        # categories centers
        for i, name_cat, col in zip(range(len(cat_list)), cat_list, c1):
            # Showing the categories centers
            ax.scatter(cent_cat_df.iloc[:, 0].loc[name_cat],
                       cent_cat_df.iloc[:, 1].loc[name_cat],
                       marker='o', c=col, alpha=0.7, s=size,
                       edgecolor='lightgrey', linewidths=edgelinesize,
                       label="{}: {} | tw={:0.2f}".format(i, name_cat,
                                                          ser_tw_cat[name_cat]),
                       zorder=10) # for the labels only
            ax.scatter(cent_cat_df.iloc[:, 0].loc[name_cat],
                       cent_cat_df.iloc[:, 1].loc[name_cat],
                       marker='o', c=col, alpha=0.7, s=centersize,
                       edgecolor='k', zorder=10) # to plot the circle
            # Showing the categories centers labels (number)
            ax.scatter(cent_cat_df.iloc[:, 0].loc[name_cat],
                       cent_cat_df.iloc[:, 1].loc[name_cat],
                       marker=r"$ {} $".format(i+1),
                       c='k', alpha=1, s=70, zorder=100)
        
        # clusters centers
        for i, name_clust, col in zip(range(len(clust_list)), clust_list, c2):
            # Showing the clusters centers
            ax.scatter(cent_clust_df.iloc[:, 0].loc[name_clust],
                       cent_clust_df.iloc[:, 1].loc[name_clust],
                       marker='o', c='lightgrey', alpha=0.7, s=size,
                       edgecolor=col, linewidths=edgelinesize,
                       label="{}: {} | tw={:0.2f}".format(i, name_clust,
                                                          ser_tw_clust[name_clust]),
                       zorder=10) # for the labels only
            ax.scatter(cent_clust_df.iloc[:, 0].loc[name_clust],
                       cent_clust_df.iloc[:, 1].loc[name_clust],
                       marker='o', c=col, alpha=0.7, s=centersize,
                       edgecolor='k', zorder=20) # to plot the circle
            # Showing the categories centers labels (number)
            ax.scatter(cent_clust_df.iloc[:, 0].loc[name_clust],
                       cent_clust_df.iloc[:, 1].loc[name_clust],
                       marker=r"$ {} $".format(i+1),
                       c='dimgrey', alpha=1, s=70, zorder=100)

        # link between the centers
        for n_cat, n_clust in zip(cat_list, clust_list):
            x1 = cent_cat_df.iloc[:,0].loc[n_cat]
            x2 = cent_clust_df.iloc[:,0].loc[n_clust]
            y1 = cent_cat_df.iloc[:,1].loc[n_cat]
            y2 = cent_clust_df.iloc[:,1].loc[n_clust]
            plt.plot([x1, x2], [y1, y2],
                     color='k', linewidth=1)

        if legend_on:
            plt.legend().get_frame().set_alpha(0.3)
            if bboxtoanchor is not None:
                plt.legend(bbox_to_anchor=bboxtoanchor)
            else: 
                plt.legend()

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
                   s=size, alpha=0.7, c='dimgrey')

    title = "Projection: " + proj + "(trustworthiness: {:.2f})".format(trustw)\
             if title is None else title
    ax.set_title(title + "\n(trustworthiness: {:.2f})".format(trustw),
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('ax 1'), ax.set_ylabel('ax 2')