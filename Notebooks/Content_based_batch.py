from sklearn.feature_extraction.text import TfidfTransformer
from Notebooks_utils.Compute_Similarity_Python import Compute_Similarity_Python
from scipy import sparse



class Content_based_recommender:
    
    def __init__(self,URM_csr, dataMatrix, topK=100, shrink = 0, normalize = True,
                 asymmetric_alpha = 0.5, tversky_alpha = 1.0, tversky_beta = 1.0,
                 similarity = "cosine", row_weights = None):
        self.TopK = topK
        self.shrink = shrink
        self.normalize = normalize
        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.dataMatrix = dataMatrix.copy()
        self.similarity = similarity
        self.row_weights = row_weights
        self.URM_csr = URM_csr
        
    def get_URM_train(self):
        return self.URM_csr
    
    def fit(self,tf_id_flag = True):
        if tf_id_flag:
            transformer = TfidfTransformer()
            transformer.fit(self.dataMatrix)
            tf_idf_icm = transformer.transform(self.dataMatrix)
            self.dataMatrix = tf_idf_icm
        similarity = Compute_Similarity_Python(self.dataMatrix, topK=self.TopK, shrink = self.shrink, normalize = self.normalize,
                 asymmetric_alpha = self.asymmetric_alpha, tversky_alpha = self.tversky_alpha, tversky_beta = self.tversky_beta,
                 similarity = self.similarity, row_weights = self.row_weights)
        self.icm_similarities = sparse.csr_matrix(similarity.compute_similarity())
    
    def recommend(self, user_ids, cutoff=10, remove_seen_flag=True, **args):
        result = []
        for user in user_ids:
            recommendation = self.single_recommendation(user, remove_seen_flag, cutoff)
            result.append(recommendation)
        print(result)
        return result
        
    def single_recommendation(self, user_id, remove_seen_flag = True, cutoff = 10):
        
        user = self.URM_csr.getrow(user_id)
        itemPopularity = user.dot(self.icm_similarities)
        popularItems = np.argsort(np.array(itemPopularity)[0])
        popularItems = np.flip(popularItems, axis = 0)

        if remove_seen_flag:
            unseen_items_mask = np.in1d(popularItems, self.URM_csr[user_id].indices,
                                        assume_unique=True, invert = True)

            unseen_items = popularItems[unseen_items_mask]
            
            recommended_items = unseen_items[0:cutoff]

        else:
            recommended_items = popularItems[0:cutoff]
            
        #recommended_items = " ".join(str(i) for i in recommended_items)
        return recommended_items
        
    
    def get_similarity(self):
        return self.icm_similarities.copy()
    
    def compute_item_score(self,user_id):
        user = self.URM_csr.getrow(user_id)
        itemPopularity = user.dot(self.icm_similarities)
        return itemPopularity