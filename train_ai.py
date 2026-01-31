# train_ai.py
import torch
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("training_data.csv")

FEATURES = [
    "ema_fast","ema_slow","rsi",
    "returns","volatility","ema_dist"
]

X = df[FEATURES].values
y = df["target"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.save")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)

model = torch.nn.Sequential(
    torch.nn.Linear(6, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
    torch.nn.Sigmoid()
)

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(150):
    preds = model(X_train)
    loss = loss_fn(preds, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 25 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.4f}")

torch.save(model.state_dict(), "ai_model.pt")
print("✅ AI model trained & saved")
