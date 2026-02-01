import torch, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("training_data.csv")

X = df[["ema9","ema21","rsi","ret","vol"]].values
y = df["target"].values.reshape(-1,1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler,"scaler.save")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

model = torch.nn.Sequential(
    torch.nn.Linear(5,32),
    torch.nn.ReLU(),
    torch.nn.Linear(32,16),
    torch.nn.ReLU(),
    torch.nn.Linear(16,1),
    torch.nn.Sigmoid()
)

opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCELoss()

for epoch in range(100):
    pred = model(X)
    loss = loss_fn(pred,y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.4f}")

torch.save(model,"ai_model.pt")
print("✅ AI retrained")
