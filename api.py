from fastapi import FastAPI
import pickle

app = FastAPI(title="ChainGuard API", description="Fraud Detection API for DeFi Transactions")

# Load precomputed risks
with open('all_risks.pkl', 'rb') as f:
    all_risks = pickle.load(f)

# Sort for top 10
all_risks_sorted = sorted(all_risks, key=lambda x: x[1], reverse=True)
top10 = all_risks_sorted[:10]

@app.get("/")
def read_root():
    return {"message": "ChainGuard Fraud Detection API"}

@app.get("/top10")
def get_top10():
    return {
        "top10": [
            {
                "index": int(idx),
                "risk_score": float(round(risk, 2)),
                "alert": alert,
                "confidence": float(round(conf, 4)),
                "predicted_label": int(label)
            } for idx, risk, alert, conf, label in top10
        ]
    }

@app.get("/risk/{index}")
def get_risk(index: int):
    if 0 <= index < len(all_risks):
        idx, risk, alert, conf, label = all_risks[index]
        return {
            "index": int(idx),
            "risk_score": float(round(risk, 2)),
            "alert": alert,
            "confidence": float(round(conf, 4)),
            "predicted_label": int(label)
        }
    else:
        return {"error": "Index out of range"}

@app.get("/stats")
def get_stats():
    high_risk = sum(1 for _, risk, _, _, _ in all_risks if risk > 80)
    medium_risk = sum(1 for _, risk, _, _, _ in all_risks if 50 < risk <= 80)
    low_risk = sum(1 for _, risk, _, _, _ in all_risks if risk <= 50)
    return {
        "total_transactions": len(all_risks),
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "low_risk": low_risk
    }