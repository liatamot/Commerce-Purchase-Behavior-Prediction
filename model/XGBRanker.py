import argparse
import os

import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import *

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="train.parquet", type=str)
    parser.add_argument("--dir_path", default="../data/", type=str)
    parser.add_argument("--output_dir", default="../output/", type=str)

    # model args
    parser.add_argument("--num_boost_round", type=int, default=100)
    parser.add_argument("--early_stopping_rounds", type=int, default=10)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)

    # train args
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    train_df = pd.read_parquet(os.path.join(args.dir_path, args.data_dir))

    user2idx = {v: k for k, v in enumerate(train_df['user_id'].unique())}
    idx2user = {k: v for k, v in enumerate(train_df['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(train_df['item_id'].unique())}
    idx2item = {k: v for k, v in enumerate(train_df['item_id'].unique())}

    # Apply the mapping functions to 'user_id' and 'item_id' columns
    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)

    # Create labels
    train_df["label"] = 1

    # Create negative samples
    user_item_pairs = set(zip(train_df['user_idx'], train_df['item_idx']))
    all_items = set(train_df['item_idx'].unique())
    negative_samples = []

    for user in train_df['user_idx'].unique():
        positive_items = set(train_df[train_df['user_idx'] == user]['item_idx'])
        negative_items = all_items - positive_items
        for item in negative_items:
            negative_samples.append((user, item, 0))

    negative_df = pd.DataFrame(negative_samples, columns=['user_idx', 'item_idx', 'label'])
    full_df = pd.concat([train_df[['user_idx', 'item_idx', 'label']], negative_df])

    # Split data
    X = full_df[['user_idx', 'item_idx']]
    y = full_df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    # Train XGBRanker model
    train_data = xgb.DMatrix(X_train, label=y_train)
    val_data = xgb.DMatrix(X_val, label=y_val)
    evals = [(train_data, 'train'), (val_data, 'eval')]

    params = {
        'objective': 'rank:pairwise',
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'eval_metric': 'ndcg',
        'seed': args.seed
    }

    model = xgb.train(params, train_data, num_boost_round=args.num_boost_round, evals=evals, early_stopping_rounds=args.early_stopping_rounds)

    # Generate recommendations
    test_users_idx = train_df['user_idx'].unique()
    recommendations = []

    for user in tqdm(test_users_idx):
        user_items = [(user, item) for item in all_items]
        user_items_df = pd.DataFrame(user_items, columns=['user_idx', 'item_idx'])
        user_items_dmatrix = xgb.DMatrix(user_items_df)
        user_items_df['score'] = model.predict(user_items_dmatrix)
        user_recommendations = user_items_df.sort_values(by='score', ascending=False).head(10)['item_idx'].values
        recommendations.extend([(user, item) for item in user_recommendations])

    sub_df = pd.DataFrame(recommendations, columns=['user_idx', 'item_idx'])
    sub_df['user_id'] = sub_df['user_idx'].map(idx2user)
    sub_df['item_id'] = sub_df['item_idx'].map(idx2item)

    outdir = args.output_dir
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    sub_df.to_csv(os.path.join(outdir, "XGBRanker_output.csv"), index=False)

if __name__ == "__main__":
    main()
