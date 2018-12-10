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


def split(train, TEST_SET_THRESHOLD=10, TEST_SET_HOLDOUT=0.25):
    """Takes train which is a pd.DataFrame, splits them into training_set, test_set which will also be pd.DataFrame"""
    grouped = train.groupby('playlist_id')['track_id'].nunique()

    clipped = grouped.index[grouped > TEST_SET_THRESHOLD].tolist()
    test_set_indices = [clipped[i] for i in
                        sorted(random.sample(range(len(clipped)), int(TEST_SET_HOLDOUT * len(clipped))))]

    test_groups = train.loc[train['playlist_id'].isin(test_set_indices)]
    test_set = pd.DataFrame(columns=["playlist_id", "track_id"])
    for name, group in test_groups.groupby('playlist_id'):
        test_set = test_set.append(group.tail(10))
    training_set = pd.concat([train, test_set, test_set]).drop_duplicates(keep=False)
    return training_set, test_set


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

        print("Recommender performance is: Precision = {:.6f}, Recall = {:.6f}, MAP = {:.6f}".format(
            cumulative_precision, cumulative_recall, cumulative_MAP))
    return cumulative_MAP
