import torch
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("training_data.csv")

X = df[
    ["ema_fast", "ema_slow", "rsi", "returns", "volatility", "ema_dist"]
].values

y = df["target"].values

# ===============================
# NORMALIZE FEATURES
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler for live trading
joblib.dump(scaler, "scaler.save")
print("✅ Scaler saved as scaler.save")

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ===============================
# MODEL (6 INPUT FEATURES)
# ===============================
model = torch.nn.Sequential(
    torch.nn.Linear(6, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1),
    torch.nn.Sigmoid()
)

# ===============================
# TRAINING SETUP
# ===============================
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ===============================
# TRAIN LOOP
# ===============================
for epoch in range(100):
    preds = model(X_train)
    loss = loss_fn(preds, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# ===============================
# EVALUATION
# ===============================
with torch.no_grad():
    preds = model(X_test)
    predicted = (preds > 0.5).float()
    accuracy = (predicted == y_test).float().mean()

print(f"\n✅ AI Accuracy: {accuracy.item()*100:.2f}%")

# ===============================
# SAVE MODEL
# ===============================
torch.save(model.state_dict(), "ai_model.pt")
print("✅ Model saved as ai_model.pt")
