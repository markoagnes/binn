from pathway_network import PathwayNetwork
import pandas as pd
import numpy as np
from binn import SparseNetworkMultiHead

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch

from data_loader import get_dataset
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from torch import nn
import torch.optim as optim

# wandb 
import wandb

batch_size = 64
learning_rate = 3e-3
num_epochs = 50

wandb = wandb.init(
    project="binn",
    config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "loss_function": "BCEWithLogitsLoss"
    }
)



obo_file_path = r'/home/markoa/workspace/data/ProstateCancer/go-basic.obo'
gaf_file_path = r'/home/markoa/workspace/data/ProstateCancer/goa_human.gaf'
protein_input_nodes_path = r'/home/markoa/workspace/data/ProstateCancer/cnv_input_genes.csv'
preprocessed_gaf_file_path = r'/home/markoa/workspace/data/ProstateCancer/goa_human_processed_filtered_8.gaf'

# Read the protein input nodes
protein_input_nodes = pd.read_csv(protein_input_nodes_path)

# Create and process the network
root_nodes_to_include = ['biological_process', 'cellular_component', 'molecular_function']
pathway_net = PathwayNetwork(obo_path=obo_file_path,
                             protein_input_nodes=protein_input_nodes,
                             root_nodes_to_include=root_nodes_to_include,
                             gaf_is_processed=True,
                             gaf_file_path=gaf_file_path,
                             preprocessed_gaf_file_path=preprocessed_gaf_file_path,
                             max_level=8,
                             verbose=True)




# Extract total nodes per level
nodes_per_level = []
for i in range(len(pathway_net.layer_indices) - 1):
    nodes_per_level.append(pathway_net.layer_indices[i+1] - pathway_net.layer_indices[i])


# Create input mapping (proteins are input)
protein_level_idx = len(nodes_per_level) - 1  # Last level is proteins
input_dim = nodes_per_level[protein_level_idx]



# Hidden layers are specific to general GO terms (excluding most general which will connect to output)
# We need to reverse since we want to go from specific to general
hidden_dims = []
for i in reversed(range(1, protein_level_idx)):
    hidden_dims.append(nodes_per_level[i])


# Set output dim to 1 for binary classification
output_dim = 1




# Create network
model = SparseNetworkMultiHead(input_dim, hidden_dims, output_dim)


# Create mapping between pathway indices and network indices
pathway_to_network_idx = {}
network_idx = 0

# Map proteins to input layer
for i in range(pathway_net.layer_indices[protein_level_idx], pathway_net.layer_indices[protein_level_idx+1]):
    pathway_to_network_idx[i] = network_idx
    network_idx += 1

# Map GO term layers from specific to general (levels 10 to 1)
for level in reversed(range(1, protein_level_idx)):
    for i in range(pathway_net.layer_indices[level], pathway_net.layer_indices[level+1]):
        pathway_to_network_idx[i] = network_idx
        network_idx += 1

# Map level 0 to the last hidden layer (connects to output)
for i in range(pathway_net.layer_indices[0], pathway_net.layer_indices[1]):
    pathway_to_network_idx[i] = network_idx
    network_idx += 1


# Convert edge indices using the mapping
sources = []
targets = []

for s, t in zip(pathway_net.edge_index[0], pathway_net.edge_index[1]):
    if s in pathway_to_network_idx and t in pathway_to_network_idx:
        sources.append(pathway_to_network_idx[s])
        targets.append(pathway_to_network_idx[t])




edge_index = torch.tensor([sources, targets])

# max index in the edge index
max_index = max(edge_index[0].max(), edge_index[1].max())
print(f"Max index in edge index: {max_index.item()}")  # Should be 9013

# total nodes in the network
total_nodes = pathway_net.graph.number_of_nodes()
print(f"Total nodes in the network: {total_nodes}")  # Should be 9013

# Set connections
model.set_connections(edge_index)



# Load dataset
features = pd.read_csv("data/ProstateCancer/pnet_index.csv", names = ["gene", "type"], header = 0)
X = np.load("data/ProstateCancer/pnet_x.npy")
y = np.load("data/ProstateCancer/pnet_y.npy")

dataset = {}
dataset['mut'] = get_dataset( "mut_important_plus_hotspots", features, X )
dataset['cnv'] = get_dataset( "cnv", features, X )




# Extract the 'cnv' data
X_cnv, features_cnv, input_genes_cnv = dataset['cnv']

y_cnv = y 
pos_count = np.sum(y_cnv == 1)
neg_count = np.sum(y_cnv == 0)


print(f"Shape of X_cnv: {X_cnv.shape}")
print(f"Shape of y_cnv: {y_cnv.shape}")
print(f"Number of input genes used: {len(input_genes_cnv)}")

added_protein_node_ids  = pathway_net.added_protein_node_ids 
print(f"Number of features to KEEP based on PathwayNetwork additions: {len(added_protein_node_ids )}") 

# Normalize the set of IDs to keep
normalized_features_to_keep = {str(item).strip().upper() for item in added_protein_node_ids}
print(f"Number of normalized features to KEEP: {len(normalized_features_to_keep)}")

id_column_name = 'gene' 

# Get the list of UniProt IDs from the correct column
uniprot_ids_from_data = features_cnv[id_column_name].tolist() 
print(f"Extracted {len(uniprot_ids_from_data)} IDs from column '{id_column_name}' in features_cnv DataFrame.") 



# Find the indices and names of features to KEEP
aligned_feature_indices = []
aligned_feature_names = [] 

# Iterate over the extracted list of UniProt IDs
for i, uniprot_id_from_list in enumerate(uniprot_ids_from_data):
    # Normalize the ID from the data
    normalized_id = str(uniprot_id_from_list).strip().upper()

    # Keep the feature if its normalized ID IS in the normalized set of added proteins
    if normalized_id in normalized_features_to_keep: 
        aligned_feature_indices.append(i)
        # Store the original ID from the list
        aligned_feature_names.append(uniprot_id_from_list) 

print(f"Number of features kept using 'added' set (with normalization): {len(aligned_feature_indices)}") # <-- Should now be 9013

# Filter the X_cnv data array using the kept indices
X_cnv_aligned = X_cnv[:, aligned_feature_indices]

print(f"Shape of aligned X_cnv: {X_cnv_aligned.shape}") # Should be [num_samples, 9013]



# Split the data
test_set_size = 0.05 
random_seed = 42    

X_train, X_test, y_train, y_test = train_test_split(
    X_cnv_aligned,
    y_cnv,
    test_size=test_set_size,
    random_state=random_seed,
    stratify=y_cnv  # IMPORTANT for classification: keeps class proportions same in train/test
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")



class CNVDataset(Dataset):
    """Custom Dataset for CNV data."""
    def __init__(self, features, labels):
        # Convert numpy arrays to PyTorch tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        # BCEWithLogitsLoss expects float targets, shape [batch_size, 1]
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1) 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    



# Create separate Datasets and DataLoaders
train_dataset = CNVDataset(X_train, y_train)
test_dataset = CNVDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# No need to shuffle the test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
print(f"Number of train batches: {len(train_loader)}, Number of test batches: {len(test_loader)}")



# Device Setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




# Define Optimizer
# optimizer = optim.Adam(model.parameters(), lr=learning_rate) 


# A better optimizer with weight decay
# optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# More advanced optimization strategy
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-4, betas=(0.9, 0.999))

# Add learning rate scheduler 
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=5, verbose=True
# )

# Warmup + cosine schedule instead of ReduceLROnPlateau
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, 
#     max_lr=learning_rate,
#     total_steps=num_epochs * len(train_loader),
#     pct_start=0.1  # 10% warmup
# )


# Training loop
model.to(device)

# pos_weight = torch.tensor([math.sqrt(neg_count/pos_count)]).to(device)  
# print(f"Class distribution - Positive: {pos_count}, Negative: {neg_count}")
# print(f"Using positive weight: {pos_weight.item():.4f}")

# Define Loss Function 
# criterion = nn.BCEWithLogitsLoss()

# if i use skip connections and averaged predictions, applied sigmoid, then the loss function is simple BCE
criterion = nn.BCELoss()


torch.autograd.set_detect_anomaly(True) 

for epoch in range(num_epochs):
    epoch_total_loss = 0.0 
    periodic_loss_sum = 0.0 

    model.train() 
    for i, (batch_features, batch_labels) in enumerate(train_loader):
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        # Ensure the labels tensor contains floating-point numbers.
        # BCEWithLogitsLoss typically expects float targets.
        # Reshape the labels tensor to have shape [batch_size, 1].
        # '.view(-1, 1)' achieves this: -1 infers the batch size dimension.
        batch_labels = batch_labels.float().view(-1, 1) 

        # Reset the gradients accumulated from the previous batch.
        # PyTorch accumulates gradients by default, so before calculating gradients
        # for the current batch, we must clear the old ones.
        optimizer.zero_grad()

        outputs = model(batch_features)

        loss = criterion(outputs, batch_labels)

        # Perform the backward pass (backpropagation)
        loss.backward()

        # Add gradient clipping 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update the network's parameters: Tell the 'optimizer'
        # to adjust the parameters based on the gradients computed in '.backward()'
        # and its internal update rule (e.g., using learning rate, momentum).
        optimizer.step()

        # Get the scalar loss value for the current batch
        current_batch_loss = loss.item() 

        # Add to both accumulators
        periodic_loss_sum += current_batch_loss
        epoch_total_loss += current_batch_loss

        # Print average loss for the last 10 batches periodically
        if (i + 1) % 10 == 0:
            avg_periodic_loss = periodic_loss_sum / 10 
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Avg Loss (last 10): {avg_periodic_loss:.4f}')

            # Log the average loss to wandb 
            batch_metrics = {
                "batch/avg_loss_10": avg_periodic_loss,
                "batch": i + epoch * len(train_loader)
            }
            wandb.log(batch_metrics)


            periodic_loss_sum = 0.0 


        

    avg_epoch_loss = epoch_total_loss / len(train_loader) 

    print(f"Epoch [{epoch+1}/{num_epochs}] Average Training Loss: {avg_epoch_loss:.4f}")

    # Update the learning rate based on validation performance
    # scheduler.step(avg_epoch_loss)

    #  log the current learning rate to wandb
    current_lr = optimizer.param_groups[0]['lr']
    wandb.log({"learning_rate": current_lr, "epoch": epoch+1})

    wandb.log({"epoch/avg_loss": avg_epoch_loss, "epoch": epoch+1})


wandb.finish() 


# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for i, (batch_features, batch_labels) in enumerate(train_loader):
#         batch_features = batch_features.to(device)
#         batch_labels = batch_labels.to(device)

#         # Ensure labels are float and have the shape [batch_size, 1]
#         batch_labels = batch_labels.float().view(-1, 1)

#         optimizer.zero_grad()
#         outputs = network(batch_features)
#         loss = criterion(outputs, batch_labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         # Print training loss periodically
#         if (i + 1) % 10 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
#             running_loss = 0.0

#     # Print average loss for the epoch
#     avg_epoch_loss = running_loss / len(train_loader)
#     print(f"Epoch [{epoch+1}/{num_epochs}] Average Training Loss: {avg_epoch_loss:.4f}")





from sklearn.metrics import accuracy_score, precision_recall_fscore_support
print("Starting evaluation on the test set...")
model.eval() 

all_labels = []
all_preds = []

# Disable gradient calculations during evaluation
with torch.no_grad(): 
    for batch_features, batch_labels in test_loader:
        # Move input features to the computation device
        batch_features = batch_features.to(device)
        # Labels are needed on CPU for sklearn metrics, can stay there or move back later

        # Forward pass through the network
        outputs = model(batch_features)

        # Apply sigmoid to convert raw outputs (logits) to probabilities
        predicted_probs = torch.sigmoid(outputs)

        # Convert probabilities to binary predictions using a 0.5 threshold
        predicted_labels = (predicted_probs > 0.5).float()

        # Collect results: move tensors to CPU and convert to NumPy arrays
        all_labels.extend(batch_labels.cpu().numpy()) 
        all_preds.extend(predicted_labels.cpu().numpy())
        # all_probs.extend(predicted_probs.cpu().numpy()) # Optional

# Check if the test loader actually yielded any data
if not all_labels:
    print("Evaluation failed: No data processed from the test loader.")
else:
    # Convert lists to NumPy arrays for scikit-learn functions
    all_labels = np.array(all_labels).flatten() 
    all_preds = np.array(all_preds).flatten()   

    print(f"Shape of all_labels: {all_labels.shape}") # Debug print
    print(f"Shape of all_preds: {all_preds.shape}")   # Debug print
    print(f"Sample labels: {all_labels[:10]}")        # Debug print
    print(f"Sample preds: {all_preds[:10]}")         # Debug print

    accuracy = accuracy_score(all_labels, all_preds)

    # Calculate precision, recall, and F1-score for the positive class (assumed to be 1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_preds, 
        average='binary', # Specify averaging for binary case
        zero_division=0
    )

    # --- Print Results ---
    print("-" * 30)
    print("Final Test Set Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print("-" * 30)






# network.eval() # Set model to evaluation mode
# all_labels = []
# all_preds = []
# with torch.no_grad(): # Disable gradient calculations
#     for batch_features, batch_labels in test_loader: # Use the test_loader
#         batch_features = batch_features.to(device)
#         # batch_labels don't need to be on GPU for sklearn metrics

#         outputs = network(batch_features)
#         predicted_probs = torch.sigmoid(outputs)
#         predicted_labels = (predicted_probs > 0.5).float()

#         all_labels.extend(batch_labels.cpu().numpy()) # Collect true labels
#         all_preds.extend(predicted_labels.cpu().numpy()) # Collect predictions

# # Ensure labels and predictions are numpy arrays
# all_labels = np.array(all_labels)
# all_preds = np.array(all_preds)

# # Calculate metrics on the TEST set
# accuracy = accuracy_score(all_labels, all_preds)
# precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

# print("-" * 30)
# print("Final Test Set Performance:")
# print(f"  Accuracy: {accuracy:.4f}")
# print(f"  Precision: {precision:.4f}")
# print(f"  Recall: {recall:.4f}")
# print(f"  F1-Score: {f1:.4f}")
# print("-" * 30)
