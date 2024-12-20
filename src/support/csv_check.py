import pandas as pd

n = 20 
df = pd.read_csv('output/qwen257b_boolq_similarities.csv')
filtered_df = df[(df['layers_to_prune'] == n) & (df['metric'] == 'exp')].sort_values('similarity', ascending=False)
print(filtered_df)

 # get starting layer index and layers to prune
start_layer = filtered_df.iloc[0]['starting_layer']
layers_to_prune = range(start_layer, start_layer + n )
layers = list(range(0,28))
   
print(f"starting layer: {start_layer}")
print(layers_to_prune)
print(layers)

for i in layers:
    if i in layers_to_prune:
        print(f"{i} prune this one")
    else:
        print(f"{i} don't prune this one")
