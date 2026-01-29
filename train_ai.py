import torch
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("training_data.csv")

X = df[[
    "ema_fast","ema_slow","rsi",
    "returns","volatility","ema_dist"
]].values

y = df["target"].values.reshape(-1,1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.save")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

model = torch.nn.Sequential(
    torch.nn.Linear(6,32),
    torch.nn.ReLU(),
    torch.nn.Linear(32,16),
    torch.nn.ReLU(),
    torch.nn.Linear(16,1),
    torch.nn.Sigmoid()
)

loss_fn = torch.nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(150):
    preds = model(X_train)
    loss = loss_fn(preds, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 25 == 0:
        print(f"Epoch {epoch} | Loss={loss.item():.4f}")

torch.save(model.state_dict(), "ai_model.pt")
print("✅ AI trained & saved")
