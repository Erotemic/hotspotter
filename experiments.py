import load_data2
import match_chips2 as mc2
from Pref import Pref


def default_preferences():
    root = Pref()
    
    pref_feat = Pref()
    pref_feat.kpts_type = Pref(2,choices=['DOG','HESS','HESAFF'])
    pref_feat.desc_type = Pref(0,choices=['SIFT'])

    pref_bow = Pref()
    pref_bow.vocab_size = 1e5

    pref_1v1 = Pref()
    
    pref_1vM = Pref()


def oxford_bag_of_words_reproduce_philbin07():
    '''
    
    
Image collection: 11 Oxford landmarks (ie particular part of a building) and distractors.

Landmark images were taken from Flickr [3], using queries such as “Oxford Christ Church” and “Oxford Radcliffe Camera.”
Distractors were taken by seaching on “Oxford” alone.
The entire dataset consists of 5,062 high resolution (1024 × 768) images.

For each landmark we chose 5 different query regions.
The ﬁve queries are used so that retrieval performance can be averaged over any individual query peculiarities.

We obtain ground truth manually by searching over the entire dataset for the 11 landmarks.

Images are assigned one of four possible labels:
    (1) Good – a nice, clear picture of the object/building.
    (2) OK – more than 25% of the object is clearly visible.
    (3) Junk – less than 25% of the object is visible, or there is a very high level of occlusion or distortion. 
    (4) Absent – the object is not present. 
    
    The number of occurrences of the different landmarks range between 7 and 220 good and ok images.
    
In addition to this labelled set, we use two other datasets to stress-test
retrieval performance when scaling up. These consist of images crawled from
Flickr’s list of most popular tags. The images in our datasets will not in
general be disjoint when crawled from Flickr, so we remove exact duplicates from the sets. 

We then assume that these datasets contain no occurrences of the objects being
searched for, so they act as distractors, testing both the performance and
scalability of our system. 


100K dataset: Crawled from Flickr's 145 most popular tags and
consists of 99,782 high resolution (1024 × 768) images.  

1M dataset. Crawled from Flickr's 450 most popular tags and
consists of 1,040,801 medium resolution (500 × 333) images.

Table 1: 
 _____________________________________________________
|Dataset # images     # features   Size of descriptors|
|-----------------------------------------------------|
|5K         5,062     16,334,970                1.9 GB|
|100K      99,782    277,770,833               33.1 GB|
|1M     1,040,801  1,186,469,709              141.4 GB|
|-----------------------------------------------------|
|Total  1,145,645  1,480,575,512              176.4 GB|
|-----------------------------------------------------|

To evaluate the performance we use the average precision (AP) measure computed
as the area under the precision-recall curve for a query. 

Precision is defined as the ratio of retrieved positive images to the total number retrieved.

Recall is defined as the ratio of the number of retrieved positive images to the total
number of positive images in the corpus. 

We compute an average precision score for each of the 5 queries for a landmark,
averaging these to obtain a mean Average Precision (mAP) score.

The average of these mAP scores is used as a single number to evaluate the
overall performance. 

In computing the average precision, we use the Good and Ok images as positive
examples of the landmark in question, Absent images as negative examples and
Junk images as null examples. These null examples are treated as though they are
not present in the database – our score is unaffected whether they are returned
or not.

'''
    OXFORD = load_data.OXFORD
    hs = mc2.HotSpotter(dbdir)
    oxford_train_cxs    = []
    oxford_test_cxs     = []
    oxford_database_cxs = []
    hs.set_train_test_database(oxford_train_cxs, 
                               oxford_test_cxs, 
                               oxford_database_cxs)
    hs.use_matcher('bagofwords')
    cx2_res = mc2.run_matching(hs)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    oxford_bag_of_words()
