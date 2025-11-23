# ChainGuard: AI/ML DeFi Transaction Fraud Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

ChainGuard is an advanced fraud detection system for decentralized finance (DeFi) transactions, leveraging graph neural networks (GNNs) to analyze blockchain data and flag anomalies. It provides risk scores, actionable alerts, and an interactive dashboard for monitoring malicious wallet addresses.

## 🚀 Features

- **Graph-Based Fraud Detection**: Ensemble of GNNs (GCN, GAT, GIN) trained on the Elliptic Bitcoin dataset for structural anomaly detection.
- **Risk Scoring & Alerts**: Generates Risk Scores (0-100) with alerts like "High Risk: Flag Wallet for Investigation".
- **Interactive Dashboard**: Streamlit UI with FastAPI backend for exploring top risky transactions, analytics, and graph visualizations.
- **Retraining Capability**: Retrain models with custom epochs directly from the UI.
- **Visualizations**: PR/ROC curves, confusion matrices, feature histograms, and interactive graph components.
- **High Performance**: F1 0.91, AP 0.96, AUC 0.99 with cost-sensitive thresholding.

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- GPU recommended (CUDA-compatible)
- 12GB+ RAM for data processing

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/devansh0703/ChainGuard-GNN.git
   cd ChainGuard-GNN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA
   pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
   pip install fastapi uvicorn streamlit plotly networkx matplotlib seaborn scikit-learn pandas
   ```

3. Download the Elliptic dataset (place in project root):
   - `elliptic_txs_classes.csv`
   - `elliptic_txs_edgelist.csv`
   - `elliptic_txs_features.csv`

## 🚀 Usage

### Training the Model
Run the training script:
```bash
python train_gin.py --epochs 100
```
This trains the GNN ensemble and saves models to `models.pt` and `scaler.pt`.

### Starting the Application
1. **Backend (FastAPI)**:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

2. **Frontend (Streamlit)**:
   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

3. Access the dashboard at `http://localhost:8501`.

### Key Features in UI
- **Overview**: Risk distribution stats.
- **Top Risky Transactions**: List of highest-risk transactions.
- **Transaction Lookup**: Search by index for details.
- **Analytics**: Model metrics and plots.
- **Graph Visualization**: Interactive views of transaction components.
- **Train Model**: Retrain with custom epochs.

## 📁 Project Structure

```
ChainGuard-GNN/
├── api.py                 # FastAPI backend
├── app.py                 # Streamlit frontend
├── train_gin.py           # Training script
├── prepare_data.py        # Data preparation
├── extract_features.py    # Feature extraction
├── add_bg_degree.py       # Background degree computation
├── add_labels.py          # Labeling script
├── models.pt              # Saved trained models
├── scaler.pt              # Saved scaler
├── graph_data.pt          # Processed graph data
├── mappings.pt            # Label mappings
├── plots/                 # Generated visualizations
├── *.csv                  # Dataset files
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 📊 Dataset

- **Source**: Elliptic Bitcoin dataset (203,769 transactions, 234,355 edges).
- **Features**: 166 per node (transaction stats, time steps).
- **Labels**: Component-level (licit/suspicious).
- **Processing**: Chunked loading for large files, graph construction, oversampling.

## 🧠 Model Architecture

- **Ensemble**: GCN, GAT, GIN with JumpingKnowledge.
- **Loss**: Focal Loss for imbalance.
- **Thresholding**: Cost-sensitive (FP=1, FN=1).
- **Evaluation**: Precision, Recall, F1, AP, AUC.

## 🔌 API Endpoints

- `GET /`: Root message.
- `GET /top10`: Top 10 riskiest transactions.
- `GET /risk/{index}`: Risk details for a transaction.
- `GET /stats`: Overall risk distribution.

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -am 'Add feature'`.
4. Push: `git push origin feature-name`.
5. Submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or issues, open an issue on GitHub or contact the maintainer.

---

**ChainGuard**: Protecting DeFi with AI-powered graph intelligence. 🚀