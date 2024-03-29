
from hscom import __common__
(print, print_, print_on, print_off, rrr, profile,
 printDBG) = __common__.init(__name__, '[classifier]', DEBUG=False)
# Science
import numpy as np
# HotSpotter
from hotspotter import report_results2 as rr2


def get_gt_cases(hs):
    valid_cxs = hs.get_valid_cxs()
    cx_list = [cx for cx in valid_cxs if len(hs.get_other_indexed_cxs(cx)) > 0]
    return cx_list


def get_labeled_descriptors(allres, orgtype_='false'):
    qcxs = allres[orgtype_].qcxs
    cxs  = allres[orgtype_].cxs
    match_list = list(zip(qcxs, cxs))
    aggdesc1, aggdesc2 = rr2.get_matching_descriptors(allres, match_list)
    return aggdesc1, aggdesc2


def train_classifier(hs):
    cx_list = get_gt_cases(hs)
    allres = rr2.get_allres(hs, cx_list)
    neg_eg = get_labeled_descriptors(allres, 'false')
    pos_eg = get_labeled_descriptors(allres, 'true')

    # Cast to a datatype we can manipulate
    neg_eg = np.array(neg_eg, np.int32)
    pos_eg = np.array(pos_eg, np.int32)

    method = 3

    if method == 1:
        # Concatentation Method
        neg_vec = np.hstack([neg_eg[0], neg_eg[1]])
        pos_vec = np.hstack([pos_eg[0], pos_eg[1]])
    elif method == 2:
        # Multiplication Method (Cao and Snavely)
        neg_vec = neg_eg[0] * neg_eg[1]
        pos_vec = pos_eg[0] * pos_eg[1]
    elif method == 3:
        # My idea
        neg_vec = np.hstack([neg_eg[0] * neg_eg[1], (neg_eg[0] - neg_eg[1]) ** 2])
        pos_vec = np.hstack([pos_eg[0] * pos_eg[1], (pos_eg[0] - pos_eg[1]) ** 2])

    pos_train = pos_vec[0:len(pos_vec) // 2]
    neg_train = neg_vec[0:len(neg_vec) // 2]
    pos_test = pos_vec[len(pos_vec) // 2:]
    neg_test = neg_vec[len(neg_vec) // 2:]
    pos_lbl = np.ones((len(pos_train), 1), dtype=np.int32)
    neg_lbl = np.zeros((len(neg_train), 1), dtype=np.int32)
    train_data = np.vstack((pos_train, neg_train))
    train_lbl = np.vstack((pos_lbl, neg_lbl))


def test_classifier(classifier, pos_test, neg_test):
    pos_output = classifier.predict(pos_test)
    neg_output = classifier.predict(neg_test)

    tp_rate = pos_output.sum() / len(pos_output)
    fp_rate = neg_output.sum() / len(neg_output)
    print('tp_rate = %r' % tp_rate)
    print('fp_rate = %r' % fp_rate)


def train_random_forest(hs, train_data, train_lbl, pos_test, neg_test):
    from sklearn.ensemble import RandomForestClassifier
    forest_parms = {
        'n_estimators': 10,
        'criterion': 'gini',
        'max_features': None,  # 'auto'
        'max_depth': None,
        'n_jobs': -1,
        'random_state': None,
        'verbose': True,
    }
    classifier = RandomForestClassifier(**forest_parms)
    classifier = classifier.fit(train_data, train_lbl)
    test_classifier(classifier, pos_test, neg_test)


def train_svm(hs, train_data, train_lbl, pos_test, neg_test):
    from sklearn import svm
    classifier = svm.SVC(C=1.0,
                         cache_size=200,
                         class_weight=None,
                         coef0=0.0,
                         degree=3,
                         gamma=0.0,
                         kernel='rbf',
                         max_iter=-1,
                         probability=False,
                         shrinking=True,
                         tol=0.001,
                         verbose=True)
    classifier.fit(train_data, train_lbl)
    test_classifier(classifier, pos_test, neg_test)
