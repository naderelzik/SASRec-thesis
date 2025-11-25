import sys
import copy
import random
import numpy as np
from collections import defaultdict
import pandas as pd
import os


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]
def data_partition_xlong(dataset_path):
    def split_sessions(seq):
        return [int(x.strip()) for x in seq.split(",")]

    def load_ratings_df(folder_path):
        train_path = os.path.join(folder_path, 'train_corpus_total_dual.txt')
        test_path = os.path.join(folder_path, 'test_corpus_total_dual.txt')
        train = pd.read_csv(train_path, header=None, sep='\t')
        test = pd.read_csv(test_path, header=None, sep='\t')
        df = pd.DataFrame(np.concatenate([train, test], axis=0))
      
        df[2] = df[2].apply(split_sessions)
        df.apply(lambda x: x[2].extend(split_sessions(str(x[3]))), axis=1)

        df = df[2].values
        df = pd.DataFrame(zip(range(len(df)), df), columns=['uid', 'sid'])
        df = df.explode('sid')
        df['uid'] = df['uid'].astype(int)
        df['sid'] = df['sid'].astype(int)
        return df

    def remove_immediate_repeats(df):
        df_next = df.shift()
        is_not_repeat = (df['uid'] != df_next['uid']) | (df['sid'] != df_next['sid'])
        df = df[is_not_repeat]
        return df

    def filter_triplets(df, min_sc=0, min_uc=0):
        print('Filtering triplets')
        if min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= min_sc]
            df = df[df['sid'].isin(good_items)]

        if min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= min_uc]
            df = df[df['uid'].isin(good_users)]
        return df

    print("Processing xlong dataset...")
    df = load_ratings_df(dataset_path)
    
    # Apply preprocessing steps
    df = remove_immediate_repeats(df)
    df = filter_triplets(df, min_sc=5, min_uc=5)  # Adjust min_sc/min_uc as needed
    
    # Re-index
    user_map = {uid: i+1 for i, uid in enumerate(df['uid'].unique())}
    item_map = {sid: i+1 for i, sid in enumerate(df['sid'].unique())}
    df['uid'] = df['uid'].map(user_map)
    df['sid'] = df['sid'].map(item_map)

    usernum = len(user_map)
    itemnum = len(item_map)

    user_train = {}
    user_valid = {}
    user_test = {}

    user2items = df.groupby('uid')['sid'].apply(list)

    for user in range(1, usernum + 1):
        items = user2items.get(user, [])
        if len(items) < 3:
            user_train[user] = items
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = items[:-2]
            user_valid[user] = [items[-2]]
            user_test[user] = [items[-1]]

    return [user_train, user_valid, user_test, usernum, itemnum]

def data_partition_foursquare(file_path):
    def load_ratings_df(path):
        df = pd.read_csv(path, sep="\t", header=None)
        df.columns = ['uid', 'venue_id', 'timestamp', 'tz_offset']
        # Sort by timestamp so sequences are chronological
        df = df.sort_values(by=['uid', 'timestamp'])
        return df[['uid', 'venue_id']]

    def remove_immediate_repeats(df):
        df_next = df.shift()
        is_not_repeat = (df['uid'] != df_next['uid']) | (df['venue_id'] != df_next['venue_id'])
        return df[is_not_repeat]

    def filter_triplets(df, min_sc=0, min_uc=0):
        print('Filtering triplets...')
        if min_sc > 0:
            item_sizes = df.groupby('venue_id').size()
            good_items = item_sizes.index[item_sizes >= min_sc]
            df = df[df['venue_id'].isin(good_items)]
        if min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= min_uc]
            df = df[df['uid'].isin(good_users)]
        return df

    print("📦 Processing Foursquare dataset...")
    df = load_ratings_df(file_path)
    
    # Remove duplicates in sequence
    df = remove_immediate_repeats(df)
    
    # Filter out inactive users/items
    df = filter_triplets(df, min_sc=5, min_uc=5)
    
    # Reindex users and items
    user_map = {uid: i+1 for i, uid in enumerate(df['uid'].unique())}
    item_map = {iid: i+1 for i, iid in enumerate(df['venue_id'].unique())}
    df['uid'] = df['uid'].map(user_map)
    df['sid'] = df['venue_id'].map(item_map)

    usernum = len(user_map)
    itemnum = len(item_map)

    # Group by user
    user2items = df.groupby('uid')['sid'].apply(list)

    # Split into train, validation, test
    user_train, user_valid, user_test = {}, {}, {}
    for user in range(1, usernum + 1):
        items = user2items.get(user, [])
        if len(items) < 3:
            user_train[user] = items
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = items[:-2]
            user_valid[user] = [items[-2]]
            user_test[user] = [items[-1]]

    print(f"✅ Done! Users: {usernum}, Items: {itemnum}, Interactions: {len(df)}")
    return [user_train, user_valid, user_test, usernum, itemnum]
def evaluate(model, dataset, args,sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    NDCG_20 = 0.0
    HT_20 = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(train[u])
        rated.add(0)
        #Negative sampling of 100
        if args.eval_mode == 'sample':
            ds_name = str(args.dataset).lower().strip()

            # Datasets where we want 100 negatives
            small_sample_datasets = {
                'ml-1m',
                'synthetic_two_patterns_no_noise',
                'synthetic_two_patterns_begin_noise',
                'synthetic_two_patterns_mid_noise',
                'synthetic_two_patterns_end_noise',
                'foursquare'
            }

            # Datasets where we want 10000 negatives
            large_sample_datasets = {
                'xlong'
            }

            if ds_name in small_sample_datasets:
                num_samples = 100
            elif ds_name in large_sample_datasets:

                num_samples = 10000
            item_idx = [test[u][0]]
            for _ in range(num_samples):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)
        elif args.eval_mode == 'full':
            #Evaluating on whole item set
            item_idx = [i for i in range(1, itemnum + 1) if i not in rated]
            item_idx.insert(0, test[u][0])  # Ensure the test item is the first item
        

        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]
        # print(f"user={u}, num_items={len(item_idx)},  predictions[:10]={predictions[:10]}")
        # print(f"  min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}, "
        #     f"nan_count={(~np.isfinite(predictions)).sum()}")

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        # Calculate Hit@10 and NDCG@10
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HT_20 += 1

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, NDCG_20/ valid_user, HT_20/ valid_user

# evaluate on val set
def evaluate_valid(model, dataset, args,sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    
    NDCG = 0.0
    valid_user = 0.0
    NDCG_20 = 0.0
    HT_20 = 0.0
    HT = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]  # Validation item
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        #print("I will calculate predictions")
        #item_idx = [i for i in range(1, itemnum + 1) if i not in rated]
        #item_idx.insert(0, valid[u][0])  # Ensure the valid item is the first item


        
        predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]
        #print("I calculated predictions")
        rank = predictions.argsort().argsort()[0].item()
        #print("I got the rank")
        valid_user += 1
        #print("I am before calculating hit and NDCG")
        # Calculate Hit@10 and NDCG@10
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HT_20 += 1
 

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, NDCG_20/valid_user, HT_20/valid_user

    
