from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fine_tuning.inference import ESGInference

try:
    global_inference_model: ESGInference = ESGInference()
except Exception as e:
    print(f"KRYTYCZNY BŁĄD podczas ładowania modelu: {e}")
    global_inference_model = None

class ReportText(BaseModel):
    text: str

app = FastAPI()

ml_to_dto_map = {
    'c1_transition_plan': 'Criteria1',
    'c2_risk_management': 'Criteria2',
    'c4_boundaries': 'Criteria4',
    'c6_historical_data': 'Criteria6',
    'c7_intensity_metrics': 'Criteria7',
    'c8_targets_credibility': 'Criteria8',
}

@app.post("/analyze_fine_tuning")
async def analyze_fine_tuning(report_data: ReportText) -> Dict[str, float]:
    if global_inference_model is None:
        raise HTTPException(status_code=503, detail="Model ML nie został zainicjowany lub wystąpił błąd podczas ładowania.")
    try:
        ml_results_detailed = global_inference_model.predict(report_data.text)
        
        dto_output = {}
        for internal_name, dto_name in ml_to_dto_map.items():
            if internal_name in ml_results_detailed:
                dto_output[dto_name] = ml_results_detailed[internal_name]["prediction"]
        
        return dto_output
    except Exception as e:
        print(f"Błąd podczas predykcji: {e}")
        raise HTTPException(status_code=500, detail=f"Wystąpił błąd podczas przetwarzania analizy: {e}")

@app.post("/analyze_RAG")
async def analyze_RAG(report_data: ReportText):
    return {"message": "Endpoint RAG jeszcze nie zaimplementowany.", "received_text_length": len(report_data.text) if report_data.text else 0}