from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
import os
from recommender import (
    get_user_analysis,
    get_svd_recommendations,
    get_item_recommendations,
    get_system_metadata,
    get_system_statistics
)

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    meta = get_system_metadata()
    stats = get_system_statistics()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "valid_ids": meta["valid_ids"],
        "valid_games": meta["valid_games"],
        "system_stats": stats
    })


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_user(request: Request, user_id: int = Form(...)):
    meta = get_system_metadata()
    stats = get_system_statistics()

    user_analysis = get_user_analysis(user_id)
    recommendations = get_svd_recommendations(user_id, n=10)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "valid_ids": meta["valid_ids"],
        "valid_games": meta["valid_games"],
        "system_stats": stats,
        "user_analysis": user_analysis,
        "recommendations": recommendations,
        "active_mode": "user"
    })


@app.post("/similar", response_class=HTMLResponse)
async def find_similar(request: Request, game_title: str = Form(...)):
    meta = get_system_metadata()
    stats = get_system_statistics()

    similar_games = get_item_recommendations(game_title, n=10)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "valid_ids": meta["valid_ids"],
        "valid_games": meta["valid_games"],
        "system_stats": stats,
        "similar_games": similar_games,
        "selected_game": game_title,
        "active_mode": "similar"
    })


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)