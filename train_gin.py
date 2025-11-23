import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, TransformerConv, JumpingKnowledge
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, average_precision_score, roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train each model')
args = parser.parse_args()
epochs = args.epochs

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Load data
data = torch.load('graph_data.pt', weights_only=False)
mappings = torch.load('mappings.pt', weights_only=False)
le = mappings['le']

# Split indices
indices = np.arange(data.num_nodes)
train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=data.y.numpy(), random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.25, stratify=data.y[train_idx].numpy(), random_state=42)  # 60% train, 20% val, 20% test

# To tensors
train_idx = torch.tensor(train_idx, dtype=torch.long)
val_idx = torch.tensor(val_idx, dtype=torch.long)
test_idx = torch.tensor(test_idx, dtype=torch.long)

# Models
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return self.classifier(x)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels))
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return self.classifier(x)

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )))
        for i in range(1, num_layers):
            if i % 3 == 1:
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            elif i % 3 == 2:
                self.convs.append(TransformerConv(hidden_channels, hidden_channels, heads=2, concat=False))
            else:
                self.convs.append(GINConv(torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.BatchNorm1d(hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels)
                )))
        self.jk = JumpingKnowledge(mode='cat')
        self.classifier = torch.nn.Linear(hidden_channels * num_layers, out_channels)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index):
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            xs.append(x)
        x = self.jk(xs)
        return self.classifier(x)

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

models = {
    'GCN': GCN(data.num_node_features, 64, len(le.classes_)).to(device),
    'GAT': GAT(data.num_node_features, 64, len(le.classes_)).to(device),
    'GIN': GIN(data.num_node_features, 64, len(le.classes_), num_layers=4).to(device)
}

optimizers = {name: torch.optim.Adam(model.parameters(), lr=0.01) for name, model in models.items()}

# Weights for imbalance
class_counts = np.bincount(data.y.cpu().numpy())
weights = 1. / class_counts
weights = torch.tensor(weights, dtype=torch.float).to(device)
# criterion = torch.nn.CrossEntropyLoss(weight=weights)
criterion = FocalLoss(alpha=1, gamma=1)

# Train each model
for name, model in models.items():
    print(f"Training {name}")
    for epoch in range(epochs):
        model.train()
        optimizers[name].zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizers[name].step()
        if epoch % 10 == 0:
            print(f'{name} Epoch {epoch}, Loss: {loss.item():.4f}')

# Get probs on val
val_probs = {}
for name, model in models.items():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out[val_idx], dim=1)
        val_probs[name] = probs.cpu().numpy()

val_y = data.y[val_idx].cpu().numpy()

# Tuned soft voting: find weights that maximize F1 for illicit (assuming 1 is illicit)
illicit_idx = 1  # Assuming le.classes_[1] is 'suspicious' or illicit

best_f1 = 0
best_weights = [1/3, 1/3, 1/3]

for w1 in np.arange(0, 1.1, 0.1):
    for w2 in np.arange(0, 1.1 - w1, 0.1):
        w3 = 1 - w1 - w2
        if w3 < 0:
            continue
        ensemble_probs = w1 * val_probs['GCN'] + w2 * val_probs['GAT'] + w3 * val_probs['GIN']
        pred = np.argmax(ensemble_probs, axis=1)
        f1 = f1_score(val_y, pred, average='macro')  # Or for illicit: f1_score(val_y == illicit_idx, pred == illicit_idx)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = [w1, w2, w3]

print(f"Best weights: GCN {best_weights[0]:.2f}, GAT {best_weights[1]:.2f}, GIN {best_weights[2]:.2f}, F1: {best_f1:.4f}")

# Test with best weights
test_probs = {}
for name, model in models.items():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out[test_idx], dim=1)
        test_probs[name] = probs.cpu().numpy()

ensemble_test_probs = best_weights[0] * test_probs['GCN'] + best_weights[1] * test_probs['GAT'] + best_weights[2] * test_probs['GIN']
pred = np.argmax(ensemble_test_probs, axis=1)
true = data.y[test_idx].cpu().numpy()

print(classification_report(true, pred, target_names=le.classes_))

# Threshold tuning with cost: FP cost 1, FN cost 3
suspicious_idx = 1
ensemble_val_probs = best_weights[0] * val_probs['GCN'] + best_weights[1] * val_probs['GAT'] + best_weights[2] * val_probs['GIN']
val_susp_prob = ensemble_val_probs[:, suspicious_idx]
val_y_binary = (val_y == suspicious_idx).astype(int)
thresholds = np.arange(0.01, 0.99, 0.01)
best_cost = float('inf')
best_thresh = 0.5
for thresh in thresholds:
    pred_thresh = (val_susp_prob > thresh).astype(int)
    FP = ((pred_thresh == 1) & (val_y_binary == 0)).sum()
    FN = ((pred_thresh == 0) & (val_y_binary == 1)).sum()
    cost = FP * 1 + FN * 1
    if cost < best_cost:
        best_cost = cost
        best_thresh = thresh
print(f"Best threshold: {best_thresh:.2f}, Cost: {best_cost}")

# Apply to test
ensemble_test_probs = best_weights[0] * test_probs['GCN'] + best_weights[1] * test_probs['GAT'] + best_weights[2] * test_probs['GIN']
test_susp_prob = ensemble_test_probs[:, suspicious_idx]
pred_soft = (test_susp_prob > best_thresh).astype(int)
pred_full = np.zeros_like(true)
pred_full[pred_soft == 1] = suspicious_idx
print("Soft Voting with Threshold:")
print(classification_report(true, pred_full, target_names=le.classes_))

# Compute metrics
y_true_binary = (true == suspicious_idx).astype(int)
ap_soft = average_precision_score(y_true_binary, test_susp_prob)
auc_soft = roc_auc_score(y_true_binary, test_susp_prob)
print(f"AP: {ap_soft:.4f}, AUC: {auc_soft:.4f}")

# Plotting
# PR Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_true_binary, test_susp_prob)
plt.figure()
plt.plot(recall_vals, precision_vals)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (AP = {:.4f})'.format(ap_soft))
plt.savefig('plots/pr_curve.png')
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true_binary, test_susp_prob)
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC = {:.4f})'.format(auc_soft))
plt.savefig('plots/roc_curve.png')
plt.close()

# Confusion Matrix
cm = confusion_matrix(true, pred_full)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('plots/confusion_matrix.png')
plt.close()

# Class Distribution
class_counts = [(data.y == i).sum().item() for i in range(len(le.classes_))]
plt.figure()
plt.bar(le.classes_, class_counts)
plt.title('Class Distribution (After Oversampling)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig('plots/class_distribution.png')
plt.close()

# Degree Distribution
degrees = data.edge_index[1].bincount().cpu().numpy()
plt.figure()
plt.hist(degrees, bins=50, log=True)
plt.title('Degree Distribution (Log Scale)')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.savefig('plots/degree_distribution.png')
plt.close()

# Threshold vs Metrics
thresholds_plot = np.arange(0.1, 0.9, 0.05)
precisions = []
recalls = []
f1s = []
for thresh in thresholds_plot:
    pred = (test_susp_prob > thresh).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true_binary, pred, average='binary')
    precisions.append(p)
    recalls.append(r)
    f1s.append(f)

plt.figure()
plt.plot(thresholds_plot, precisions, label='Precision')
plt.plot(thresholds_plot, recalls, label='Recall')
plt.plot(thresholds_plot, f1s, label='F1')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Metrics vs Threshold')
plt.legend()
plt.savefig('plots/threshold_metrics.png')
plt.close()

# Component sizes
cc_df = pd.read_csv('connected_components.csv')
cc_sizes = cc_df['ccId'].value_counts()
plt.figure()
plt.hist(cc_sizes, bins=50, log=True)
plt.title('Connected Component Sizes (Log Scale)')
plt.xlabel('Size')
plt.ylabel('Frequency')
plt.savefig('plots/component_sizes.png')
plt.close()

# Feature histograms (first 5 features)
nodes_df = pd.read_csv('nodes_with_bg_degree.csv')
for i in range(1, 6):
    feat = f'feat#{i}'
    if feat in nodes_df.columns:
        plt.figure()
        plt.hist(nodes_df[feat], bins=50)
        plt.title(f'Histogram of {feat}')
        plt.xlabel(feat)
        plt.ylabel('Frequency')
        plt.savefig(f'plots/{feat}_hist.png')
        plt.close()

# Small graph sample
# Removed sample graph as it's not the full real graph

# In-degree vs Out-degree scatter
in_degrees = data.edge_index[1].bincount().cpu().numpy()
out_degrees = data.edge_index[0].bincount().cpu().numpy()
plt.figure()
plt.scatter(out_degrees, in_degrees, alpha=0.5, s=1)
plt.xlabel('Out-Degree')
plt.ylabel('In-Degree')
plt.title('In-Degree vs Out-Degree Scatter')
plt.xscale('log')
plt.yscale('log')
plt.savefig('plots/degree_scatter.png')
plt.close()

# Feature correlation heatmap (first 10 features)
feat_cols = [f'feat#{i}' for i in range(1, 11) if f'feat#{i}' in nodes_df.columns]
if feat_cols:
    corr = nodes_df[feat_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap (First 10 Features)')
    plt.savefig('plots/feature_correlation.png')
    plt.close()

print("Plots saved in plots/ directory.")

# Risk Scorer and Action Logic
def get_risk_score(confidence, label):
    return confidence * 100

def get_alert(risk_score):
    if risk_score > 80:
        return "High Risk: Flag Wallet for Investigation"
    elif risk_score > 50:
        return "Medium Risk: Add Sender to Watchlist"
    else:
        return "Low Risk: Normal Transaction"

# For test set, compute risk scores
test_risks = []
for i in range(len(test_idx)):
    conf = test_susp_prob[i]
    label = pred_full[i]
    risk = get_risk_score(conf, label)
    alert = get_alert(risk)
    test_risks.append((i, risk, alert, conf, label))

# Top 10 riskiest
test_risks.sort(key=lambda x: x[1], reverse=True)
print("Top 10 Riskiest Transactions:")
for i, (idx, risk, alert, conf, label) in enumerate(test_risks[:10]):
    print(f"{i+1}. Index: {idx}, Risk Score: {risk:.2f}, Alert: {alert}, Confidence: {conf:.4f}, Label: {label}")

# Bottom 10 lowest risk
test_risks.sort(key=lambda x: x[1])  # sort ascending
print("Bottom 10 Lowest Risk Transactions:")
for i, (idx, risk, alert, conf, label) in enumerate(test_risks[:10]):
    print(f"{i+1}. Index: {idx}, Risk Score: {risk:.2f}, Alert: {alert}, Confidence: {conf:.4f}, Label: {label}")

# Save models
torch.save(models, 'models.pt')
torch.save({'scaler': mappings['scaler']}, 'scaler.pt')

# Compute all predictions
all_probs = {}
for name, model in models.items():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)
        all_probs[name] = probs.cpu().numpy()

ensemble_all_probs = best_weights[0] * all_probs['GCN'] + best_weights[1] * all_probs['GAT'] + best_weights[2] * all_probs['GIN']
all_susp_prob = ensemble_all_probs[:, suspicious_idx]
all_pred_thresh = (all_susp_prob > best_thresh).astype(int)
all_labels = data.y.cpu().numpy()

all_risks = []
for i in range(data.num_nodes):
    conf = all_susp_prob[i]
    label = all_pred_thresh[i]
    risk = get_risk_score(conf, label)
    alert = get_alert(risk)
    all_risks.append((i, risk, alert, conf, label))

# Save all_risks
import pickle
with open('all_risks.pkl', 'wb') as f:
    pickle.dump(all_risks, f)

print("Models, scaler, and risks saved.")