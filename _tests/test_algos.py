'''
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    from hsviz import draw_func2 as df2
    np.random.seed(seed=0)  # RANDOM SEED (for reproducibility)
    is_whiten = helpers.get_flag('--whiten')
    dim = helpers.get_arg('--dim', type_=int, default=3)
    K = helpers.get_arg('--K', type_=int, default=10)
    num_clusters = K
    __REAL_DATA_MODE__ = True
    if __REAL_DATA_MODE__:
        import main
        hs = main.main(defaultdb='NAUTS')
        cache_dir = hs.dirs.cache_dir
        cx2_desc = hs.feats.cx2_desc
        data = np.vstack(cx2_desc)
    else:
        cache_dir = 'akmeans_test'
        data = np.random.rand(1000, 3)
    datax2_clusterx, clusters = precompute_akmeans(data, num_clusters,
                                                   force_recomp=False,
                                                   cache_dir=cache_dir)
    fig = plot_clusters(data, datax2_clusterx, clusters, num_pca_dims=dim,
                        whiten=is_whiten)
    fig.show()
    exec(df2.present())
#IDEA:
#intead have each datapoint "pull" on one another. Maybe warp the space
#in which they sit with a covariance matrix.  basically let gravity do
#the clustering.  Check to see if any algos like this.

#itertools.groupby
#groups
'''
