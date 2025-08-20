from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import PersonData
from app.model_utils import predict_fitness
import os

app = FastAPI(title="Fitness Prediction API")

# Static & templates
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok"}


# ---- API JSON Endpoint ----
@app.post("/predict", response_class=JSONResponse)
def predict_api(data: PersonData):
    """Accept JSON requests"""
    try:
        result = predict_fitness(data)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ---- Web Form Endpoint ----
@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    age: float = Form(...),
    weight: float = Form(...),
    height: float = Form(...),
    gender: str = Form(...),
):
    """Accept HTML form submissions"""
    # Convert to schema-compatible dict
    gender_m = 1 if gender.lower() == "male" else 0

    payload = PersonData(
        age=age,
        height_cm=height,
        weight_kg=weight,
        heart_rate=70,           # default placeholder
        blood_pressure=120,      # default placeholder
        sleep_hours=7,           # default placeholder
        nutrition_quality=5,     # default placeholder
        activity_index=5,        # default placeholder
        smokes=0,                # default placeholder
        gender_M=gender_m,
    )

    result = predict_fitness(payload)

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "prediction": result},
    )
