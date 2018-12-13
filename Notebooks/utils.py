import pandas as pd
import numpy as np
from scipy import sparse
import random


def classify_durations(data):
    data.loc[data['duration_sec'].isin(range(60)), 'duration_sec'] = 1
    data.loc[data['duration_sec'].isin(range(60, 120)), 'duration_sec'] = 2
    data.loc[data['duration_sec'].isin(range(120, 180)), 'duration_sec'] = 3
    data.loc[data['duration_sec'].isin(range(180, 240)), 'duration_sec'] = 4
    data.loc[data['duration_sec'].isin(range(240, 300)), 'duration_sec'] = 5
    data.loc[data['duration_sec'].isin(range(300, 200000)), 'duration_sec'] = 6

    data['duration_sec'].value_counts()


def build_icm_csr(data):
    
    classify_durations(data)
    
    albums_id = data['album_id']
    artists_id = data['artist_id']
    durations = data['duration_sec']
    tracks = data['track_id']

    albums_max = np.amax(albums_id)
    artists_max = np.amax(artists_id)
    durations_max = np.amax(durations)
    number_of_songs = data.shape[0]

    icm_csr_matrix = sparse.csr_matrix((number_of_songs, albums_max + artists_max + durations_max + 3),
                                       dtype=np.uint32)

    icm_csr_matrix[tracks, albums_id] = 1
    icm_csr_matrix[tracks, albums_max + artists_id] = 1
    icm_csr_matrix[tracks, albums_max + artists_max + durations] = 1

    return icm_csr_matrix


def build_urm_csr(data):
    fill_data = np.ones(data.shape[0])
    # posso usare gli id direttamente solo perchè come già detto sono consistenti
    row = data['playlist_id'].values
    col = data['track_id'].values
    n_pl = np.amax(data['playlist_id']) + 1
    n_tr = np.amax(data['track_id']) + 1

    return sparse.csr_matrix((fill_data, (row, col)), dtype=np.float32, shape=(50446,20635))


def build_csv(items):
    recommended_items = " ".join(str(i) for i in items)
    return recommended_items

def build_train_target_nn(training_set):
    
    occurrencies = training_set.getnnz(axis = 1)
    mask = np.where(occurrencies > 10)
    training_set_clean = training_set[mask]
    
    target_set = sparse.csr_matrix(training_set_clean.shape, dtype = np.float32)

    for item in range(training_set_clean.shape[0]):
        user = training_set_clean.getrow(item).indices
        selection = np.random.choice(user, size = 10, replace = False)
        training_set_clean[item, selection] = 0
        target_set[item, selection] = 1
        
    training_set_clean.eliminate_zeros()
    return training_set_clean, target_set

def split(URM_csr, TEST_SET_THRESHOLD=10, TEST_SET_HOLDOUT=0.25):
    """Takes an URM_csr, splits them into training_set, test_set which will also are URM_csr """
    nnz_per_row = URM_csr.getnnz(axis=1)
    result = np.where(nnz_per_row > TEST_SET_THRESHOLD)[0]
    test_mask = np.random.choice([True,False], len(result),p = [TEST_SET_HOLDOUT, 1 - TEST_SET_HOLDOUT])
    URM_train = URM_csr.copy()
    URM_test = sparse.csr_matrix(URM_csr.shape,dtype=np.float32)
    for i in result[test_mask]:
        test_sample = URM_csr.getrow(i)
        nnz_in_test_sample = test_sample.indices
        test_samples = np.random.choice(nnz_in_test_sample,TEST_SET_THRESHOLD,replace=False)
        chosen_mask = np.zeros(20635,dtype=bool)
        chosen_mask[test_samples] = True
        URM_train[i,chosen_mask] = 0
        URM_test[i,chosen_mask] = 1
    URM_train.eliminate_zeros()
    return URM_train, URM_test



def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(recommended_items)

    return precision_score


def evaluate_algorithm(URM_test, recommender_object, target_playlists, at=10):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    for user_id in target_playlists:
        target_items = URM_test.getrow(user_id).indices

        recommended_items = recommender_object.recommend(user_id, at=at)
        num_eval += 1

        cumulative_precision += precision(recommended_items, target_items)
        cumulative_recall += recall(recommended_items, target_items)
        cumulative_MAP += MAP(recommended_items, target_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval
        
    return cumulative_MAP

