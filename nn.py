import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class DenseNet(nn.Module):
    def __init__(self, input_dim, hidden_size1=256, hidden_size2=128, hidden_size3=64, hidden_size4=32,
                 dropout_rate1=0.3, dropout_rate2=0.3, dropout_rate3=0.3, dropout_rate4=0.2):
        super(DenseNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate1),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate2),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Dropout(dropout_rate3),
            nn.Linear(hidden_size3, hidden_size4),
            nn.ReLU(),
            nn.Dropout(dropout_rate4),
            nn.Linear(hidden_size4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train(data, epochs=10, hyperparams=None):
    data.groupby('status').sample(n=2185, replace=True)

    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]

    y = y.apply(lambda x: 1 if x == 'COVID-19' else 0)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train_tensor = torch.FloatTensor(x_train)
    x_test_tensor = torch.FloatTensor(x_test)
    y_train_tensor = torch.FloatTensor(y_train.values)
    y_test_tensor = torch.FloatTensor(y_test.values)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    batch_size = hyperparams['batch_size'] if hyperparams else 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_dim = x_train.shape[1]
    
    if hyperparams:
        model = DenseNet(input_dim,
                        hidden_size1=hyperparams['hidden_size1'],
                        hidden_size2=hyperparams['hidden_size2'],
                        hidden_size3=hyperparams['hidden_size3'],
                        hidden_size4=hyperparams['hidden_size4'],
                        dropout_rate1=hyperparams['dropout_rate1'],
                        dropout_rate2=hyperparams['dropout_rate2'],
                        dropout_rate3=hyperparams['dropout_rate3'],
                        dropout_rate4=hyperparams['dropout_rate4'])
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    else:
        model = DenseNet(input_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.BCELoss()

    print('Training model...')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            labels = labels.view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

    torch.save(model.state_dict(), 'model.pth')

    print('Evaluating model...')
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.view(-1, 1)

            outputs = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_pred_prob, y_true = np.array(all_predictions), np.array(all_labels)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = y_true.astype(int)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def eval(data, uuid):
    df = data[data['uuid'] == uuid]

    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    y = y.apply(lambda x: 1 if x == 'COVID-19' else 0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y.values)

    dataset = TensorDataset(X_tensor, y_tensor)

    model = DenseNet(X.shape[1])
    model.load_state_dict(torch.load('model.pth'))

    model.eval()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataset):
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            print('Segment', idx, 'Predicted:', predicted.item(), 'Actual:', labels.item())

