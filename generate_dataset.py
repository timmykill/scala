import pandas as pd
import numpy as np

num_users = 20
num_items = 7

std_dev = 2
size = 1000

df = pd.DataFrame(np.zeros((num_users, num_items)), columns=[
                  f'Item_{i}' for i in range(num_items)])

for user_id in range(num_users):
    preferred_item = user_id % num_items
    data = np.random.normal(preferred_item, std_dev, size)
    # round to nearest integer
    data = np.rint(data).astype(int)
    # remove invalid values
    data = data[data >= 0]
    data = data[data < num_items]
    # get how many times each item was chosen
    data = np.bincount(data)
    # scale values to be between 1 and 5
    data = data / np.max(data) * 4 + 1
    # two decimals is enough
    data = np.round(data, 2)

    for item_id in range(num_items):
        df.iloc[user_id, item_id] = data[item_id] or 1

# print the dataframe
df.index.name = 'User_ID'
df.columns.name = 'Items'
print(df.head())

# save data in a file with the following format:
#  user_id::item_id::rating

with open('dataset.txt', 'w') as f:
    for user_id in range(num_users):
        for item_id in range(num_items):
            f.write(f'{user_id}::{item_id}::{df.iloc[user_id, item_id]}\n')
