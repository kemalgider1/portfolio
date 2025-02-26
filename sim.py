import pandas as pd
# Read the original similarity file
df = pd.read_csv('similarity_file_old.csv')
# Pivot the file so that each cluster becomes a separate column with its similarity score
df_wide = df.pivot(index='IATA', columns='CLUSTER_IATA', values='SIMILARITY_SCORE').reset_index()
# Optionally, if your code expects a column named 'Cluster', you might combine the values into one column:
# df_wide['Cluster'] = df_wide.apply(lambda row: some_logic(row), axis=1)
df_wide.to_csv('similarity_file.csv', index=False)