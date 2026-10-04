import os
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from ml_engine import MLEngine

app = FastAPI(
    title="Smart Hospital Appointment & Patient Management System",
    description="Data Analytics, EDA Visualizations, Data Preprocessing, and ML Patient No-Show Prediction API",
    version="1.1.0"
)

ml_engine = MLEngine()

class PredictionRequest(BaseModel):
    Age: float = 40.0
    Gender: str = "Female"
    Department: str = "General Medicine"
    Lead_Days: float = 7.0
    Hypertension: int = 0
    Diabetes: int = 0
    Alcoholism: int = 0
    Previous_No_Shows: int = 0
    SMS_Reminder: int = 1

@app.get("/api/analytics")
def get_analytics():
    return {
        "eda": ml_engine.eda_data,
        "preprocessing": ml_engine.preprocessing_summary
    }

@app.get("/api/preprocessing")
def get_preprocessing_details():
    return ml_engine.preprocessing_summary

@app.get("/api/models/compare")
def compare_models():
    return {
        "metrics": ml_engine.metrics
    }

@app.post("/api/predict")
def predict_no_show(req: PredictionRequest):
    result = ml_engine.predict_no_show(req.model_dump())
    return result

@app.get("/api/patients")
def get_patients(
    search: Optional[str] = None,
    dept: Optional[str] = None,
    no_show: Optional[int] = None,
    limit: int = 50,
    offset: int = 0
):
    df = ml_engine.clean_df.copy()

    if search:
        search_lower = search.lower()
        df = df[
            df["Patient_ID"].str.lower().str.contains(search_lower) |
            df["Department"].str.lower().str.contains(search_lower)
        ]

    if dept and dept != "All":
        df = df[df["Department"] == dept]

    if no_show is not None:
        df = df[df["No_Show"] == no_show]

    total_records = len(df)
    paged_df = df.iloc[offset : offset + limit]

    records = paged_df.to_dict(orient="records")
    return {
        "total": total_records,
        "offset": offset,
        "limit": limit,
        "records": records
    }

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
