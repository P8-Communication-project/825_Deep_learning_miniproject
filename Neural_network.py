import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import torch
import wandb
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix

# Flag for outputting confusion matrix:
CONFMATRIX = True

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# GPU turned out to be quite slow in our case, so we recommend using CPU.
device = "cpu"
#wandb.init(project='your_project_name', config=Config(your_custom_config_dict))

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #print(type(self.data[idx]), type(self.labels[idx]))
        return torch.tensor(self.data[idx], dtype=torch.float).to(device), torch.tensor([self.labels[idx]], dtype=torch.float).to(device)

# -------------------------------------------------------------------
# --- uncomment the code below to use the different layer-models: ---
# -------------------------------------------------------------------

# ------ 6 layers: ------
# class BinaryClassifier(nn.Module):
#     def __init__(self, input_size, dropout_p=0):
#         super(BinaryClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_size, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 256)
#         self.fc4 = nn.Linear(256, 128)
#         self.fc5 = nn.Linear(128, 64)
#         self.fc6 = nn.Linear(64, 1)
#         self.dropout = nn.Dropout(dropout_p)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         x = self.dropout(torch.relu(self.fc1(x)))
#         x = self.dropout(torch.relu(self.fc2(x)))
#         x = self.dropout(torch.relu(self.fc3(x)))
#         x = self.dropout(torch.relu(self.fc4(x)))
#         x = self.dropout(torch.relu(self.fc5(x)))
#         x = self.sigmoid(self.fc6(x))
#         return x
# -----------------------

# ------ 5 layers: ------
# class BinaryClassifier(nn.Module):
#     def __init__(self, input_size, dropout_p=0):
#         super(BinaryClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_size, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 64)
#         self.fc5 = nn.Linear(64, 1)
#         self.dropout = nn.Dropout(dropout_p)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         x = self.dropout(torch.relu(self.fc1(x)))
#         x = self.dropout(torch.relu(self.fc2(x)))
#         x = self.dropout(torch.relu(self.fc3(x)))
#         x = self.dropout(torch.relu(self.fc4(x)))
#         x = self.sigmoid(self.fc5(x))
#         return x
# -----------------------
    
# ------ 4 layers: ------
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, dropout_p=0):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout_p)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))
        return x
# -----------------------


# Load the dataset
orig_df = pd.read_csv('undersampled_data_with_transform.csv')
aug_df = pd.read_csv('augmented_undersampled_data_with_transform.csv')


# Convert the 'transformed_data' column to a tensor
# Note: You may need to preprocess the 'transformed_data' column to convert it from a string to a list of floats

x_orig = orig_df['transformed_data'].apply(lambda z: list(map(float, z.strip("tensor(").strip("[[").strip('\n').strip("]])").split(','))))
x_noisy = aug_df['noise'].apply(lambda z: list(map(float, z.strip("tensor(").strip("[[").strip('\n').strip("]])").split(','))))
x_pitch_up = aug_df['pitch_up'].apply(lambda z: list(map(float, z.strip("tensor(").strip("[[").strip('\n').strip("]])").split(','))))
x_pitch_down = aug_df['pitch_down'].apply(lambda z: list(map(float, z.strip("tensor(").strip("[[").strip('\n').strip("]])").split(','))))


#Encode healthy and covid status as 1 and 0
for status in orig_df["status"]:
    if status == 'healthy':
        orig_df['status'] = orig_df['status'].replace('healthy', 1)
    else:
        orig_df['status'] = orig_df['status'].replace('COVID-19', 0)
        
# Repeat the labels three times since they are the same for the augmented data
orig_df = pd.concat([orig_df]*4, ignore_index=True)
y = torch.tensor(orig_df['status'].values)
print(f"Labels: {y.shape}")

# Concatenate the original and augmented data
x_combined = torch.tensor(pd.concat([x_orig, x_noisy, x_pitch_up, x_pitch_down],ignore_index=True)
                          .tolist(),
                          dtype=torch.float)

print(f"data tensor: {x_combined.shape}")


# Create data loaders for training, test and validation:
# doing the splits separately for the original and augmented data, using the same seed to be able to reference the same UUIDs later. 
def build_dataset(batch_size):
    X_train1, X_temp1, y_train1, y_temp1 = train_test_split(x_orig, y[:len(x_orig)], test_size=0.3, random_state=42)
    X_val1, X_test1, y_val1, y_test1 = train_test_split(X_temp1, y_temp1, test_size=0.33, random_state=42)
    
    #Repeat this for the augmented data
    X_train2, X_temp2, y_train2, y_temp2 = train_test_split(x_noisy, y[len(x_orig):2*len(x_orig)], test_size=0.3, random_state=42)
    X_val2, X_test2, y_val2, y_test2 = train_test_split(X_temp2, y_temp2, test_size=0.33, random_state=42)
    
    X_train3, X_temp3, y_train3, y_temp3 = train_test_split(x_pitch_up, y[2*len(x_orig):3*len(x_orig)], test_size=0.3, random_state=42)
    X_val3, X_test3, y_val3, y_test3 = train_test_split(X_temp3, y_temp3, test_size=0.33, random_state=42)
    
    X_train4, X_temp4, y_train4, y_temp4 = train_test_split(x_pitch_down, y[3*len(x_orig):], test_size=0.3, random_state=42)
    X_val4, X_test4, y_val4, y_test4 = train_test_split(X_temp4, y_temp4, test_size=0.33, random_state=42)

    X_train_UUID1, X_temp_uuid, y_train5, y_temp5 = train_test_split(aug_df["uuid"], y[:len(x_orig)], test_size=0.3, random_state=42)
    X_val_uuid, X_test5, y_val5, y_test5 = train_test_split(X_temp_uuid, y_temp5, test_size=0.33, random_state=42)
    
    #Concatenate all x_train and y_train
    X_train = torch.tensor([*X_train1,*X_train2,*X_train3,*X_train4])
    y_train = torch.tensor([*y_train1,*y_train2,*y_train3,*y_train4])
    
    #Concatenate all x_val and y_val
    X_val = torch.tensor([*X_val1,*X_val2,*X_val3,*X_val4])
    y_val = torch.tensor([*y_val1,*y_val2,*y_val3,*y_val4])
    X_val_uuid_total = [*X_val_uuid, *X_val_uuid, *X_val_uuid, *X_val_uuid]
    
    #Concatenate all x_test and y_test
    X_test = torch.tensor([*X_test1,*X_test2,*X_test3,*X_test4])
    y_test = torch.tensor([*y_test1,*y_test2,*y_test3,*y_test4]) 
    
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)    
    
    #Count number of ones in y_train and y_val
    print(f"Number of ones in y_train: {torch.sum(y_train)}, size of y_train: {y_train.size()}")
    print(f"Number of ones in y_val: {torch.sum(y_val)}, size of y_val: {y_val.size()}")

    train_dataset= CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(y_val), X_val_uuid_total

def train(config=None):
    # Intialize wandb
    with wandb.init(config=config):
        # Initialize the model
        config = wandb.config
        #print(config)

        train_loader, val_loader, y_val_len, uuids_to_val = build_dataset(config.batch_size)
        
        model = BinaryClassifier(input_size=768, dropout_p=config.dropout_p)
        model.to(device)
        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
        criterion = nn.BCELoss()

        # Logging metrics during training
        for epoch in range(config.epochs):
            # Training loop
            model.train()  # Set the model to training mode
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Zero the gradients
                #print("Input size: ", inputs.size())
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update the weights
                
                # Log metrics to wandb
            wandb.log({'train_loss': loss.item(), 'epoch': epoch})

            # Validation loop
            model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            correct = 0
            total = 0
            predicted_list = np.array([])
            labels_list = np.array([])
        
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels)
                    predicted = torch.round(outputs)
                    total += labels.size(0)
                    predicted_list = np.append(predicted_list, predicted.cpu().numpy())
                    labels_list = np.append(labels_list, labels.cpu().numpy())

                    for predictions, label in zip(predicted, labels):
                        if predictions == label:
                            correct += 1
                # Compute validation metrics
                #print(correct, total)
                accuracy = correct / total
                val_loss = val_loss/val_loader.__len__()
                
                # Log metrics to wandb
                wandb.log({'val_loss': val_loss, 'val_accuracy': accuracy, 'epoch': epoch, "user ": "Anders"})
                # wandb.log({'val_loss': val_loss, 'val_accuracy': accuracy, 'epoch': epoch, "user ": "David"})
                # wandb.log({'val_loss': val_loss, 'val_accuracy': accuracy, 'epoch': epoch, "user ": "Mads"})
                # wandb.log({'val_loss': val_loss, 'val_accuracy': accuracy, 'epoch': epoch, "user ": "Nicolai"})

        if CONFMATRIX == True:
            print(predicted_list.shape)
            cf_matrix = confusion_matrix(labels_list, predicted_list)
            print(len(labels_list))
            # Build confusion matrix
            classes = ('COVID-19', 'Healthy')
            df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
                                index = [i for i in classes],
                                columns = [i for i in classes])
            plt.figure(figsize = (12,7))
            ax = sn.heatmap(df_cm, annot=True,)
            ax.collections[0].set_clim(0,1)
            plt.title('Confusion matrix 4 layers')
            plt.savefig('119_4_layer+1.png')

        five_random_task = True
        if five_random_task == True:
            false_positives = []
            false_negatives = []
            correct = []
            for idx, pred in enumerate(predicted_list):

                # True positive
                if pred == 1 and labels_list[idx] == 1:
                    print(f"True positive: {uuids_to_val[idx]}")
                    correct.append(uuids_to_val[idx])
                # False positive
                if pred == 1 and labels_list[idx] == 0:
                    print(f"False positive: {uuids_to_val[idx]}")
                    false_positives.append(uuids_to_val[idx])

                # False negative
                if pred == 0 and labels_list[idx] == 1:
                    print(f"False negative: {uuids_to_val[idx]}")
                    false_negatives.append(uuids_to_val[idx])

            dic = {'False positives': false_positives, 'False negatives': false_negatives, 'Correct': correct}
            import pickle
            with open('false_pos_neg.pkl', 'wb') as f:
                pickle.dump(dic, f)

        with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels)
                    predicted = torch.round(outputs)
                    total += labels.size(0)
                    predicted_list = np.append(predicted_list, predicted.cpu().numpy())
                    labels_list = np.append(labels_list, labels.cpu().numpy())

                    for predictions, label in zip(predicted, labels):
                        if predictions == label:
                            correct += 1
                # Compute validation metrics
                #print(correct, total)
                accuracy = correct / total
                val_loss = val_loss/val_loader.__len__()
                
                # Log metrics to wandb
                wandb.log({'val_loss': val_loss, 'val_accuracy': accuracy, 'epoch': epoch, "user ": "Anders"})
        
# WandDB stads for logging
wandb.login()
# Initialize wandb with your project name and optionally specify other configurations

# Define sweep configuration
sweep_configuration = {
    "name": "4 layers 119 for plot",
    "method": "grid",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "learning_rate": {'values':[0.0003875] },#list(np.linspace(0.0005,0.005,5))},
        "batch_size": {"values": [32]},#[16, 32, 64]},
        "epochs": {"values": [150]},#[20, 50, 100, 150]},
        "hidden_layers":{'value': 4},
        "dropout_p":{'values': [0.5]},#[0, 0.2, 0.5]},
        "optimizer": {"values": ["adam"]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='825-miniproject-DL')
#sweep_id = '825-miniproject-DL/825-miniproject-DL/wkcqgpfu'
#print("sweepid: ", sweep_id)
wandb.agent(sweep_id, function=train)