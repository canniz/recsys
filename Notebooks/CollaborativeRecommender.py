import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
import utils
from Notebooks_utils import Compute_Similarity_Python as sim

class CollaborativeItemBasedRecommender(object):
    
    def fit(self, URM_csr, block_size = 100, **args):

        similarity_object = sim.Compute_Similarity_Python(URM_csr, **args)
        
        self.item_similarities = similarity_object.compute_similarity(block_size = block_size)
        self.URM_csr = URM_csr
        
    def get_URM_train(self):
        return self.URM_csr
    
    def saveModel(path, file_name):
        print("NON STO SALVANDO IN " + str(path) + " PERCHE' E' UN CAZZO DI TEST")
    
    def recommend(self, user_id, cutoff=10, remove_seen = True, **args):
        
        user = self.URM_csr.getrow(user_id)
        itemPopularity = user.dot(self.item_similarities)
        popularItems = np.argsort(np.array(itemPopularity.todense())[0])
        popularItems = np.flip(popularItems, axis = 0)

        if remove_seen:
            unseen_items_mask = np.in1d(popularItems, self.URM_csr[user_id].indices,
                                        assume_unique=True, invert = True)

            unseen_items = popularItems[unseen_items_mask]
            
            recommended_items = unseen_items[0:cutoff]

        else:
            recommended_items = popularItems[0:cutoff]
        
        return recommended_items
    
    def compute_item_score(self, user_id):
        user = self.URM_csr.getrow(user_id)
        itemPopularity = user.dot(self.item_similarities)
        
        return itemPopularity