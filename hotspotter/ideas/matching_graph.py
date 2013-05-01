N = 2000 #min_image_threshold
Beta = 1.5 # 

def matchgraph(hs):
    cm, vm = hs.get_managers("cm","vm")
    # Load Images 
    cx_list = cm.all_valid_cxs()
    num_cx = len(cx_list) # number of chips (> 10,000)

    # Train Default Bag-of-Words Model
    V = vm.train_model(cx_list) # The Set of Bag-of-Words Vectors (normalized, tf-idf preweighted)
    if len(V) < 2:
        raise Exception('Cannot build matchgraph')

    # Preallocate Intermediate Variables
    dims = len(V[0]) # dimensionality of bag of words  (>1,000,000,000)
    W = eye(len(cx_list)) # The learned weighting of word histogram similarity     
    Sim = np.zeros((num_cx,num_cx), dtype=uint8)
    svm_train_examples = np.zeros((num_cx,num_cx, dims), dtype=float)
    # Perform Batch Query
    for x_b, a in enumerate(V): # a = query vector
        for x_a, b in enumerate(V): # b = database vector
            svm_train_examples[x_a, x_b, :] = qvec * dvec # LEARN!
            Sim[x_a, x_b] = np.transpose(qvec).dot(W).dot(dvec)
        tops_x = Sim[x_a, :].argsort()[::,-1] # sorted indexes
        spatial_rerank(Sim, tops_x)

    # Train SVM 
    wT = np.transpose(w)
    def hinge_loss(y, x, w):
        val = 1 - y * wT.dot(x)
        return max(0, val)

    def svm_cost(y, x, w, C):
        from sklearn import svm
        # C is regularization variable
        training_pairs = rand.select(cx_list, 'pairwise', replacement=False)
        .5 * wT.dot(w) + C * sum( [hinge_loss(y[a,b], x[a,b], w) for (a,b) in training_pairs ] )**2

        C      = param(1, min=0, max=inf)
        kernel = param('''Specifies the kernel type to be used in the algorithm.
                       It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
                       ‘precomputed’ or a callable. If none is given, ‘rbf’ will
                       be used. If a callable is given it is used to precompute
                       the kernel matrix.''', 
                        'linear', choices=['linear','poly','rbf','sigmoid','precomputed'])

        degree = param('''Degree of kernel function. It is significant only in
                       ‘poly’ and ‘sigmoid''',
                       3, sigifeq=(kernel,['poly','sigmoid']))

        gamma  = param('''Kernel coefficient for ‘rbf’ and ‘poly’. If gamma is
                       0.0 then 1/n_features will be used instead.''',
                       0, sigifeq=(kernel,['poly', rbf]))

        coef0  = param('''Independent term in kernel function. It is only
                       significant in ‘poly’ and ‘sigmoid’.''',
                       0, sigifeq=(kernel,['poly','sigmoid']))

        probability = param('''Whether to enable probability estimates. This
                            must be enabled prior to calling predict_proba.''',
                            False)

        tol = param(''' Tolerance for stopping criterion.''', 1e-3)

        shrinking = param(True, 'use shrinking heuristic')

        cache_size  = param('', 0)

        class_weight = param('''Set the parameter C of class i to
                             class_weight[i]*C for SVC. If not given, all
                             classes are supposed to have weight one. The ‘auto’
                             mode uses the values of y to automatically adjust
                             weights inversely proportional to class
                             frequencies.''', 
                             'auto', {dict, 'auto'})

        max_iter = param('''Hard limit on iterations within solver, or -1 for
                         no limit.''', -1)

        import numpy as np
        X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        y = np.array([1, 1, 2, 2])
        from sklearn.svm import SVC
        clf = SVC()
        clf.fit(X, y) 
        SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
                shrinking=True, tol=0.001, verbose=False)
        print(clf.predict([[-0.8, -1]]))

        '''
        http://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html
        decision_function(X)	    Distance of the samples X to the separating hyperplane.
        fit(X, y[, sample_weight])	Fit the SVM model according to the given training data.
        get_params([deep])			Get parameters for the estimator
        predict(X)					Perform classification on samples in X.
        predict_log_proba(X)		Compute log probabilities of possible outcomes for samples in X.
        predict_proba(X)			Compute probabilities of possible outcomes for samples in X.
        score(X, y)					Returns the mean accuracy on the given test data and labels.
        set_params(**params)		Set the parameters of the estimator.
        '''        

        svm.SVC( C=1, kernel=kernel()

    # 3.2 Iterative Learning and Matching
    # w = with vanilla tf-idf Rank[a,b]
    # While: True
    #     Compute pairwise similarity of all images using weights w
    #     Foreach image rerank a shortlist of its most similar matches
    #     if OPTION_1: 
    #        train_data, train_labels = 
    #     Learn w using Linear SVM
    # 
    # Two Stratagies: 
    # Match images with similarity above some threshold
    # Match images to their #1 Rank



    w = minimize(  )

    # Get Pairwise Matches
    qres_list = vm.batch_query(cx_list, method="TFIDF")
    qres_list.matching
    MatchingGraph = np.matrix(2,3)
    for res in qres_list:
        qcx = res.qcx
        fm.



    
