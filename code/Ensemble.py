from collections import defaultdict
from tqdm import tqdm

result = []

def get_items(df, interval=5):
    items = df['item_id'].values
    return items[::interval]

weighted_dataframes = [
    (df, weight) for df, weight in zip(dataframes, weights)
]

# 각 유저에 대해 가중치를 계산하고 상위 10개의 아이템 선택
for user_id in tqdm(users):
    temp = defaultdict(float)
    
    for df, weight in weighted_dataframes:
        items = get_items(df)
        
        for item in items:
            temp[item] += weight

    # temp에서 가중치가 높은 순서대로 상위 10개의 아이템 선택
    top_items = sorted(temp.items(), key=lambda x: x[1], reverse=True)[:10]
    for item_id, _ in top_items:
        result.append((user_id, item_id))

result_df = pd.DataFrame(result, columns=["user_id", "item_id"])
result_df.to_csv("ensemble_output.csv", index=False)
