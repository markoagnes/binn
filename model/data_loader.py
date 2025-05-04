import numpy as np
import pandas as pd
import os


# features = pd.read_csv("data/ProstateCancer/pnet_index.csv", names = ["gene", "type"], header = 0)
# X = np.load("data/ProstateCancer/pnet_x.npy")
# y = np.load("data/ProstateCancer/pnet_y.npy")
 
def get_dataset( type : str, features_in, X_in ):
    feature_indices = features_in["type"] == type
    features_in = features_in[feature_indices].reset_index(drop=True)
    X_in = X_in[:,feature_indices]
 
    # TODO: Currently the uniprot version is not unique, I use the average
 
    features_in['group'] = features_in.agg('-'.join, axis=1)
    groups = features_in['group']
    unique_groups = groups.drop_duplicates(keep='first').reset_index(drop=True)
    group_to_index = {group: idx for idx, group in enumerate(unique_groups)}
    features_in['group_index'] = groups.map(group_to_index)
 
    array_combined = np.zeros((X_in.shape[0], len(unique_groups)))
    for group_idx in range(len(unique_groups)):
        mask = (features_in['group_index'] == group_idx).to_numpy()
        array_combined[:, group_idx] = X_in[:, mask].mean(axis=1)
 
    features_in = features_in.drop_duplicates(subset=["gene", "type"], keep='first').drop(columns=['group', 'group_index']).reset_index(drop=True)
    X_in = array_combined
    print( f"Shape of X: {X_in.shape}" )
 
    input_genes = list(features_in['gene'].unique())
    print( f"Count of input genes: {len(input_genes)}" )
 
    return (X_in, features_in, input_genes)
 
# dataset = {}
# dataset['mut'] = get_dataset( "mut_important_plus_hotspots", features, X )
# dataset['cnv'] = get_dataset( "cnv", features, X )


# output_dir = "data/ProstateCancer" # Set the output directory as requested

# # Process and save 'mut' genes
# # Check if 'mut' key exists, its value is valid, has 3 elements, and the gene list is not empty
# if 'mut' in dataset and dataset['mut'] and len(dataset['mut']) > 2 and dataset['mut'][2]:
#     mut_genes_list = dataset['mut'][2] # Get the third element (index 2)
#     mut_genes_df = pd.DataFrame(mut_genes_list, columns=['gene']) # Use 'gene' as column header
#     mut_csv_path = os.path.join(output_dir, 'mut_input_genes.csv')
#     mut_genes_df.to_csv(mut_csv_path, index=False)
#     print(f"Saved 'mut' input genes to: {mut_csv_path}")
# else:
#     print("Could not find 'mut' input genes in dataset to save.")

# # Process and save 'cnv' genes
# # Check if 'cnv' key exists, its value is valid, has 3 elements, and the gene list is not empty
# if 'cnv' in dataset and dataset['cnv'] and len(dataset['cnv']) > 2 and dataset['cnv'][2]:
#     cnv_genes_list = dataset['cnv'][2] # Get the third element (index 2)
#     cnv_genes_df = pd.DataFrame(cnv_genes_list, columns=['gene']) # Use 'gene' as column header
#     cnv_csv_path = os.path.join(output_dir, 'cnv_input_genes.csv')
#     cnv_genes_df.to_csv(cnv_csv_path, index=False)
#     print(f"Saved 'cnv' input genes to: {cnv_csv_path}")
# else:
#     print("Could not find 'cnv' input genes in dataset to save.")