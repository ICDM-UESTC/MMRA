import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def compute_normalized_inner_product_similarity(vector1, vector2):

    vector1 = np.array(vector1)

    vector2 = np.array(vector2)
    
    # Here the consine similarity is the normalized inner product between two vectors.

    similarity = cosine_similarity([vector1], vector2)

    return (similarity[0]).tolist()


def sort_and_take_top_k(id_list, similarity_list, k):

    zipped_lists = list(zip(similarity_list, id_list))

    sorted_lists = sorted(zipped_lists, key=lambda x: x[0], reverse=True)

    similarity_list, id_list = zip(*sorted_lists)

    similarity_list = similarity_list[:k]

    id_list = id_list[:k]

    return list(id_list), list(similarity_list)


def retrieve_items(mode, k, train_path, valid_path, test_path):

    if mode == 'test':

        train_df = pd.read_pickle(train_path)

        valid_df = pd.read_pickle(valid_path)

        test_df = pd.read_pickle(test_path)

        database_df = pd.concat([train_df, valid_df], axis=0)

        database_df = database_df.reset_index(drop=True)

        retrieved_item_id_list = []

        retrieved_item_similarity_list = []

        for i in tqdm(range(len(test_df))):

            test_vec = test_df['retrieval_feature'][i]

            database_matrix = (database_df['retrieval_feature']).tolist()

            similarity_list = compute_normalized_inner_product_similarity(test_vec, database_matrix)

            id_list = (database_df['item_id']).tolist()

            id_list_top_k, similarity_list_top_k = sort_and_take_top_k(id_list, similarity_list, k)

            retrieved_item_id_list.append(id_list_top_k)

            retrieved_item_similarity_list.append(similarity_list_top_k)

    elif mode == 'valid':

        train_df = pd.read_pickle(train_path)

        valid_df = pd.read_pickle(valid_path)

        database_df = pd.concat([train_df, valid_df], axis=0)

        database_df = database_df.reset_index(drop=True)

        retrieved_item_id_list = []

        retrieved_item_similarity_list = []

        for i in tqdm(range(len(valid_df))):

            valid_vec = valid_df['retrieval_feature'][i]

            current_database_df = database_df[database_df['item_id'] != valid_df['item_id'][i]]

            current_database_df = current_database_df.reset_index(drop=True)

            database_matrix = (current_database_df['retrieval_feature']).tolist()

            similarity_list = compute_normalized_inner_product_similarity(valid_vec, database_matrix)

            id_list = (current_database_df['item_id']).tolist()

            id_list_top_k, similarity_list_top_k = sort_and_take_top_k(id_list, similarity_list, k)

            retrieved_item_id_list.append(id_list_top_k)

            retrieved_item_similarity_list.append(similarity_list_top_k)

    elif mode == 'train':

        train_df = pd.read_pickle(train_path)

        valid_df = pd.read_pickle(valid_path)

        database_df = pd.concat([train_df, valid_df], axis=0)

        database_df = database_df.reset_index(drop=True)

        retrieved_item_id_list = []

        retrieved_item_similarity_list = []

        for i in tqdm(range(len(train_df))):

            train_vec = train_df['retrieval_feature_2'][i]

            current_database_df = database_df[database_df['item_id'] != train_df['item_id'][i]]

            current_database_df = current_database_df.reset_index(drop=True)

            database_matrix = (current_database_df['retrieval_feature_2']).tolist()

            similarity_list = compute_normalized_inner_product_similarity(train_vec, database_matrix)

            id_list = (current_database_df['item_id']).tolist()

            id_list_top_k, similarity_list_top_k = sort_and_take_top_k(id_list, similarity_list, k)

            retrieved_item_id_list.append(id_list_top_k)

            retrieved_item_similarity_list.append(similarity_list_top_k)

    return retrieved_item_id_list, retrieved_item_similarity_list


def main(mode, k, train_path, valid_path, test_path):

    if mode == 'test':

        test_df = pd.read_pickle(test_path)

        retrieved_item_id_list, retrieved_item_similarity_list = retrieve_items(mode, k, train_path, valid_path,
                                                                                test_path)

        test_df['retrieved_item_id_list'] = retrieved_item_id_list

        test_df['retrieved_item_similarity_list'] = retrieved_item_similarity_list

        test_df.to_pickle(test_path)

        print('test retrieval finished')

    elif mode == 'valid':

        valid_df = pd.read_pickle(valid_path)

        retrieved_item_id_list, retrieved_item_similarity_list = retrieve_items(mode, k, train_path, valid_path,
                                                                                test_path)

        valid_df['retrieved_item_id_list'] = retrieved_item_id_list

        valid_df['retrieved_item_similarity_list'] = retrieved_item_similarity_list

        valid_df.to_pickle(valid_path)

        print('valid retrieval finished')

    elif mode == 'train':

        train_df = pd.read_pickle(train_path)

        retrieved_item_id_list, retrieved_item_similarity_list = retrieve_items(mode, k, train_path, valid_path,
                                                                                test_path)

        train_df['retrieved_item_id_list'] = retrieved_item_id_list

        train_df['retrieved_item_similarity_list'] = retrieved_item_similarity_list

        train_df.to_pickle(train_path)

        print('train retrieval finished')

    else:
        raise Exception('mode error')


def stack_retrieved_feature(train_path, valid_path, test_path):

    df_train = pd.read_pickle(train_path)

    df_test = pd.read_pickle(test_path)

    df_valid = pd.read_pickle(valid_path)

    df_database = pd.concat([df_train, df_test, df_valid], axis=0)

    df_database.reset_index(drop=True, inplace=True)

    retrieved_visual_feature_embedding_cls_list = []

    retrieved_visual_feature_embedding_mean_list = []

    retrieved_textual_feature_embedding_list = []

    retrieve_label_list = []

    for i in tqdm(range(len(df_train))):

        id_list = df_train['retrieved_item_id_list'][i]

        current_retrieved_visual_feature_embedding_cls_list = []

        current_retrieved_visual_feature_embedding_mean_list = []

        current_retrieved_textual_feature_embedding_list = []

        current_retrieved_label_list = []

        for j in range(len(id_list)):

            item_id = id_list[j]

            index = df_database[df_database['item_id'] == item_id].index[0]

            current_retrieved_visual_feature_embedding_cls_list.append(
                df_database['visual_feature_embedding_cls'][index])

            current_retrieved_visual_feature_embedding_mean_list.append(
                df_database['visual_feature_embedding_mean'][index])

            current_retrieved_textual_feature_embedding_list.append(df_database['textual_feature_embedding'][index])

            current_retrieved_label_list.append(df_database['label'][index])

        retrieved_visual_feature_embedding_cls_list.append(current_retrieved_visual_feature_embedding_cls_list)

        retrieved_visual_feature_embedding_mean_list.append(current_retrieved_visual_feature_embedding_mean_list)

        retrieved_textual_feature_embedding_list.append(current_retrieved_textual_feature_embedding_list)

        retrieve_label_list.append(current_retrieved_label_list)

    df_train['retrieved_visual_feature_embedding_cls'] = retrieved_visual_feature_embedding_cls_list

    df_train['retrieved_visual_feature_embedding_mean'] = retrieved_visual_feature_embedding_mean_list

    df_train['retrieved_textual_feature_embedding'] = retrieved_textual_feature_embedding_list

    df_train['retrieved_label'] = retrieve_label_list

    df_train.to_pickle(train_path)

    retrieved_visual_feature_embedding_cls_list = []

    retrieved_visual_feature_embedding_mean_list = []

    retrieved_textual_feature_embedding_list = []

    retrieve_label_list = []

    for i in tqdm(range(len(df_test))):

        id_list = df_test['retrieved_item_id_list'][i]

        current_retrieved_visual_feature_embedding_cls_list = []

        current_retrieved_visual_feature_embedding_mean_list = []

        current_retrieved_textual_feature_embedding_list = []

        current_retrieved_label_list = []

        for j in range(len(id_list)):

            item_id = id_list[j]

            index = df_database[df_database['item_id'] == item_id].index[0]

            current_retrieved_visual_feature_embedding_cls_list.append(
                df_database['visual_feature_embedding_cls'][index])

            current_retrieved_visual_feature_embedding_mean_list.append(
                df_database['visual_feature_embedding_mean'][index])

            current_retrieved_textual_feature_embedding_list.append(df_database['textual_feature_embedding'][index])

            current_retrieved_label_list.append(df_database['label'][index])

        retrieved_visual_feature_embedding_cls_list.append(current_retrieved_visual_feature_embedding_cls_list)

        retrieved_visual_feature_embedding_mean_list.append(current_retrieved_visual_feature_embedding_mean_list)

        retrieved_textual_feature_embedding_list.append(current_retrieved_textual_feature_embedding_list)

        retrieve_label_list.append(current_retrieved_label_list)

    df_test['retrieved_visual_feature_embedding_cls'] = retrieved_visual_feature_embedding_cls_list

    df_test['retrieved_visual_feature_embedding_mean'] = retrieved_visual_feature_embedding_mean_list

    df_test['retrieved_textual_feature_embedding'] = retrieved_textual_feature_embedding_list

    df_test['retrieved_label'] = retrieve_label_list

    df_test.to_pickle(test_path)

    retrieved_visual_feature_embedding_cls_list = []

    retrieved_visual_feature_embedding_mean_list = []

    retrieved_textual_feature_embedding_list = []

    retrieve_label_list = []

    for i in tqdm(range(len(df_valid))):

        id_list = df_valid['retrieved_item_id_list'][i]

        current_retrieved_visual_feature_embedding_cls_list = []

        current_retrieved_visual_feature_embedding_mean_list = []

        current_retrieved_textual_feature_embedding_list = []

        current_retrieved_label_list = []

        for j in range(len(id_list)):
            item_id = id_list[j]

            index = df_database[df_database['item_id'] == item_id].index[0]

            current_retrieved_visual_feature_embedding_cls_list.append(
                df_database['visual_feature_embedding_cls'][index])

            current_retrieved_visual_feature_embedding_mean_list.append(
                df_database['visual_feature_embedding_mean'][index])

            current_retrieved_textual_feature_embedding_list.append(df_database['textual_feature_embedding'][index])

            current_retrieved_label_list.append(df_database['label'][index])

        retrieved_visual_feature_embedding_cls_list.append(current_retrieved_visual_feature_embedding_cls_list)

        retrieved_visual_feature_embedding_mean_list.append(current_retrieved_visual_feature_embedding_mean_list)

        retrieved_textual_feature_embedding_list.append(current_retrieved_textual_feature_embedding_list)

        retrieve_label_list.append(current_retrieved_label_list)

    df_valid['retrieved_visual_feature_embedding_cls'] = retrieved_visual_feature_embedding_cls_list

    df_valid['retrieved_visual_feature_embedding_mean'] = retrieved_visual_feature_embedding_mean_list

    df_valid['retrieved_textual_feature_embedding'] = retrieved_textual_feature_embedding_list

    df_valid['retrieved_label'] = retrieve_label_list

    df_valid.to_pickle(valid_path)


if __name__ == "__main__":

    train_path = r'train.pkl'

    valid_path = r'valid.pkl'

    test_path = r'test.pkl'

    main('train', 20, train_path, valid_path, test_path)

    main('valid', 20, train_path, valid_path, test_path)

    main('test', 20, train_path, valid_path, test_path)

    stack_retrieved_feature(train_path, valid_path, test_path)
