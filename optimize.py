import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from nn import DenseNet

def objective(trial, X, y):
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate1 = trial.suggest_float('dropout_rate1', 0.1, 0.5)
    dropout_rate2 = trial.suggest_float('dropout_rate2', 0.1, 0.5)
    dropout_rate3 = trial.suggest_float('dropout_rate3', 0.1, 0.5)
    dropout_rate4 = trial.suggest_float('dropout_rate4', 0.1, 0.3)
    
    hidden_size1 = trial.suggest_int('hidden_size1', 128, 512)
    hidden_size2 = trial.suggest_int('hidden_size2', 64, 256)
    hidden_size3 = trial.suggest_int('hidden_size3', 32, 128)
    hidden_size4 = trial.suggest_int('hidden_size4', 16, 64)
    
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    epochs = trial.suggest_int('epochs', 5, 20)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Convert to tensors
    x_train_tensor = torch.FloatTensor(x_train)
    x_test_tensor = torch.FloatTensor(x_test)
    y_train_tensor = torch.FloatTensor(y_train.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model with optimized architecture
    input_dim = x_train.shape[1]
    model = DenseNet(input_dim, 
                    hidden_size1=hidden_size1,
                    hidden_size2=hidden_size2,
                    hidden_size3=hidden_size3,
                    hidden_size4=hidden_size4,
                    dropout_rate1=dropout_rate1,
                    dropout_rate2=dropout_rate2,
                    dropout_rate3=dropout_rate3,
                    dropout_rate4=dropout_rate4)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            labels = labels.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                labels = labels.view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_loss

def optimize_hyperparameters(data, n_trials=100):
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    y = y.apply(lambda x: 1 if x == 'COVID-19' else 0)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return trial.params 