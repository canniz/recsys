import numpy as np
import pandas as pd
from scipy import sparse 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity



class CF_CBF_Ensemble(object):
    
    def fit(self, URM_csr, ICM_csr, alfa):

        transformer = TfidfTransformer()
        transformer.fit(URM_csr)
        tf_idf_csr = transformer.transform(URM_csr)

        IRM = sparse.csr_matrix(tf_idf_csr.transpose())
        
        csr_similarities = sparse.csr_matrix(cosine_similarity(IRM, dense_output=False))
        

        transformer.fit(ICM_csr)
        tf_idf_icm = transformer.transform(ICM_csr)
        icm_similarities = sparse.csr_matrix(cosine_similarity(tf_idf_icm, dense_output=False))
        
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