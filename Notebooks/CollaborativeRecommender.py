import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
import utils
from Notebooks_utils import Compute_Similarity_Python as sim

class CollaborativeItemBasedRecommender(object):
    
    def fit(self, URM_csr, block_size = 100, **args):

        transformer = TfidfTransformer()
        transformer.fit(URM_csr)
        tf_idf_csr = transformer.transform(URM_csr)

        IRM = sparse.csr_matrix(tf_idf_csr.transpose())
        
        similarity_object = sim.Compute_Similarity_Python(IRM, **args)
        
        self.item_similarities = similarity_object.compute_similarity(block_size = block_size)
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
        
        return recommended_items