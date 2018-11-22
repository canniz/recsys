import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer

TEST_SET_THRESHOLD = 10
TEST_SET_HOLDOUT = 0.2
BEST_ALFA = 0.92

#Ora passiamo training_set e test_set a csr_matrix
test_set_csr = sparse.load_npz("TEST_SET_CSR.npz")
icm_csr = sparse.load_npz("NEW_ICM_CSR.npz")
urm_csr = sparse.load_npz("URM_CSR.npz")
test_set_playlists = np.unique(test_set_csr.nonzero()[0])

def precision(recommended_items, relevant_items):
    
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    
    return precision_score

def recall(recommended_items, relevant_items):
    
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    
    return recall_score

def MAP(recommended_items, relevant_items):
       
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def evaluate_algorithm(URM_test, recommender_object, target_playlists, at=10):
    
    
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0
    
    num_eval = 0


    result = []
    
    for user_id in target_playlists:
    
        target_items = URM_test.getrow(user_id).indices
        
        recommended_items = recommender_object.recommend(user_id, at=at)
        num_eval+=1
        
        cumulative_precision += precision(recommended_items, target_items)
        cumulative_recall += recall(recommended_items, target_items)
        cumulative_MAP += MAP(recommended_items, target_items)
        
        recommendation_string = " ".join(str(i) for i in recommended_items)
        temp = [user_id,recommendation_string]
        result.append(temp)


    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval
    
    rec = pd.DataFrame(result)
    rec.to_csv("sample_submission.csv", index = False, header = ["playlist_id", "track_ids"])
    print("Recommender performance is: Precision = {:.6f}, Recall = {:.6f}, MAP = {:.6f}".format(cumulative_precision, cumulative_recall, cumulative_MAP))
    


class EnsembleRecommender(object):
    
    def fit(self, URM_csr, ICM_csr, alfa):

        transformer = TfidfTransformer()
        transformer.fit(URM_csr)
        tf_idf_csr = transformer.transform(URM_csr)

        IRM = sparse.csr_matrix(tf_idf_csr.transpose())
        
        csr_similarities = sparse.csr_matrix(cosine_similarity(IRM, dense_output=False), dtype= np.float32)
        

        transformer.fit(ICM_csr)
        tf_idf_icm = transformer.transform(ICM_csr)
        icm_similarities = sparse.csr_matrix(cosine_similarity(tf_idf_icm, dense_output=False), dtype= np.float32)
        
        print("COMPUTING ENSEMBLE SIMILARITIES")
        self.item_similarities = alfa*csr_similarities + (1-alfa)*icm_similarities        
        self.URM_csr = URM_csr
        
    
    def recommend(self, user_id, at=10, remove_seen=True):
        
        user = self.URM_csr.getrow(user_id)
        itemPopularity = user.dot(self.item_similarities)
        popularItems = np.argsort(np.array(itemPopularity.todense())[0])
        popularItems = np.flip(popularItems, axis = 0)

        if remove_seen:
            unseen_items_mask = np.in1d(popularItems, self.URM_csr[user_id].indices,
                                        assume_unique=True, invert = True)

            unseen_items = popularItems[unseen_items_mask]
            
            recommended_items = unseen_items[0:at]

        else:
            recommended_items = popularItems[0:at]
            
        #recommended_items = " ".join(str(i) for i in recommended_items)
        return recommended_items


ensemble = EnsembleRecommender()
print("FITTING WITH NEW ICM")
ensemble.fit(urm_csr,icm_csr, alfa = BEST_ALFA)
print("EVALUATING...")
evaluate_algorithm(test_set_csr, ensemble, test_set_playlists)
    