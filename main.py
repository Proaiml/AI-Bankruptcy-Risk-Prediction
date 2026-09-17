import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import matplotlib.pyplot as plt
torch.set_default_device("cuda")

_orig_from_numpy = torch.from_numpy
torch.from_numpy = lambda x: _orig_from_numpy(x).cuda()


data = pd.read_csv("./data.csv")
data_x = data.drop("Bankrupt?",axis=1)
data_y = data['Bankrupt?']

X_train,X_test,Y_train,Y_test = train_test_split(data_x,data_y,test_size=0.33,random_state=42)

class MultiCell(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        k1 = self.layer1(x)
        k2 = torch.relu(k1)

        return k2




model = nn.Sequential(
    MultiCell(95,30),
    MultiCell(30,20),
    nn.Linear(20,1)
)


loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr=0.01)


X_train = torch.from_numpy(X_train.to_numpy())
Y_train = torch.from_numpy(Y_train.to_numpy())
X_test = torch.from_numpy(X_test.to_numpy())
Y_test = torch.from_numpy(Y_test.to_numpy())

X_train = X_train.float()
Y_train = Y_train.float().unsqueeze(1)
X_test = X_test.float()
Y_test = Y_test.float().unsqueeze(1)
training_loss = []
validation_loss =[]
for epoch in range(1,40):
    print(epoch)
    pred = model(X_train)
    loss = loss_fn(pred,Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    training_loss.append(loss.item())

    with torch.no_grad():
        val_pred = model(X_test)
        val_losss = loss_fn(val_pred,Y_test)

    validation_loss.append(val_losss.item())


plt.plot(training_loss, label="Train Loss")
plt.plot(validation_loss, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.show()
torch.save(model.state_dict(), "model_weights.pth")

# İlk katmanın ağırlıkları
weights = model[0].layer1.weight.detach().cpu().numpy()

plt.figure(figsize=(14,6))
plt.imshow(weights, aspect="auto")
plt.colorbar(label="Weight")

plt.xlabel("Input Feature (95 adet)")
plt.ylabel("Neuron (30 adet)")
plt.title("1. Katman Öğrenilmiş Ağırlıkları")

plt.show()