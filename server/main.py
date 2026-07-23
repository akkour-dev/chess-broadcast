"""
Serveur de diffusion d'échiquier physique.

Lancer avec :
    uvicorn main:app --host 0.0.0.0 --port 8000

Deux rôles de client, sur la même page HTML :
  - "caméra"     -> WebSocket /ws/broadcast : envoie les frames + la calibration
  - "spectateur" -> WebSocket /ws/viewer    : reçoit position (FEN) + PGN en direct

Un seul plateau suivi à la fois (variable globale `tracker`). Pour gérer
plusieurs échiquiers en parallèle, il suffirait de garder un dict de
BoardTracker par identifiant de partie.
"""

import base64
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse

from vision import BoardTracker

app = FastAPI(title="Chess Board Broadcast")

tracker = BoardTracker()
viewers: list[WebSocket] = []

CLIENT_HTML = Path(__file__).parent.parent / "client" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def index():
    return CLIENT_HTML.read_text(encoding="utf-8")


@app.get("/pgn", response_class=PlainTextResponse)
async def get_pgn():
    return str(tracker.game)


@app.get("/fen", response_class=PlainTextResponse)
async def get_fen():
    return tracker.board.fen()


@app.post("/reset")
async def reset():
    tracker.reset_game()
    await broadcast_to_viewers({"status": "reset", "fen": tracker.board.fen(), "pgn": str(tracker.game)})
    return {"ok": True}


def decode_frame(b64_jpeg: str) -> np.ndarray:
    raw = base64.b64decode(b64_jpeg.split(",")[-1])
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


async def broadcast_to_viewers(payload: dict):
    dead = []
    for ws in viewers:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in viewers:
            viewers.remove(ws)


@app.websocket("/ws/broadcast")
async def ws_broadcast(websocket: WebSocket):
    """Le client 'caméra' se connecte ici : il envoie soit un message de
    calibration, soit des frames vidéo en continu."""
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg["type"] == "calibrate":
                tracker.set_corners(msg["corners"], msg["width"], msg["height"])
                await websocket.send_text(json.dumps({"status": "calibrated"}))
                continue

            if msg["type"] == "reset":
                tracker.reset_game()
                await broadcast_to_viewers({"status": "reset", "fen": tracker.board.fen(), "pgn": str(tracker.game)})
                continue

            if msg["type"] == "frame":
                frame = decode_frame(msg["data"])
                if frame is None:
                    continue
                result = tracker.process_frame(frame)
                if result:
                    await websocket.send_text(json.dumps(result))
                    if result.get("status") == "move":
                        await broadcast_to_viewers(result)

    except WebSocketDisconnect:
        pass


@app.websocket("/ws/viewer")
async def ws_viewer(websocket: WebSocket):
    """Les spectateurs se connectent ici en lecture seule."""
    await websocket.accept()
    viewers.append(websocket)
    # état initial pour un spectateur qui arrive en cours de partie
    await websocket.send_text(json.dumps({
        "status": "sync",
        "fen": tracker.board.fen(),
        "pgn": str(tracker.game),
    }))
    try:
        while True:
            await websocket.receive_text()  # pas de messages attendus, on garde juste la connexion ouverte
    except WebSocketDisconnect:
        if websocket in viewers:
            viewers.remove(websocket)
