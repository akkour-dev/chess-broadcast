# Échiquier en direct — webcam → PGN → diffusion

Diffuse une partie d'échecs jouée sur un vrai échiquier : une webcam filme
le plateau, un serveur détecte les coups joués et les valide avec
`python-chess`, puis diffuse en direct la position et la notation PGN à
n'importe quel navigateur connecté.

## Comment ça marche

1. **Calibration** : sur la page, tu cliques les 4 coins du plateau dans
   la vidéo (a8, h8, h1, a1). Le serveur redresse l'image en une grille 8x8.
2. **Détection des coups** : pas de reconnaissance de pièce par IA ici —
   le serveur compare l'image de chaque case entre deux instants stables.
   Les cases qui ont changé = le coup joué. On confronte cet ensemble de
   cases à tous les coups légaux possibles (`python-chess`) pour trouver
   le bon coup (gère roque, prise en passant, promotion en dame par défaut).
3. **Diffusion** : chaque coup validé est poussé en PGN et renvoyé en
   direct, via WebSocket, à tous les spectateurs connectés sur `/ws/viewer`.

## Installation

```bash
cd server
python3 -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Puis ouvre `http://localhost:8000` dans un navigateur :
- Choisis **"Filmer la partie"** sur l'appareil relié à la webcam posée
  au-dessus de l'échiquier.
- Choisis **"Regarder la partie"** sur n'importe quel autre appareil du
  même réseau (`http://<ip-du-serveur>:8000`) pour suivre en direct.

Le PGN de la partie est aussi téléchargeable à tout moment sur `/pgn`.

## Réglages à ajuster selon ta configuration

Dans `server/vision.py` :
- `MOTION_THRESHOLD` : sensibilité pour détecter "quelque chose bouge
  encore" (main du joueur). Trop bas = ne se stabilise jamais.
- `CHANGE_THRESHOLD` : seuil pour dire qu'une case a vraiment changé
  entre deux positions stables. À augmenter si l'éclairage est instable
  et déclenche de faux positifs.
- `STABLE_FRAMES` : nombre de frames identiques avant de considérer la
  position comme figée (donc le coup terminé).

## Limites connues et pistes d'amélioration

- **Pas de reconnaissance du type de pièce** : la détection se base sur
  "quelles cases ont changé", pas "quelle pièce est où". Ça suffit pour
  suivre une partie légale coup par coup, mais ne permet pas de repartir
  d'une position arbitraire déjà en cours (il faut calibrer puis démarrer
  d'une partie standard).
- **Promotion** : toujours supposée en Dame. Pour gérer les autres cas,
  ajouter un petit sélecteur dans l'UI qui envoie la pièce choisie au
  serveur au moment du coup.
- **Passage à un vrai classifieur** : pour reconnaître réellement les
  pièces (utile pour démarrer depuis une position non standard, ou
  détecter les erreurs de placement), remplacer `_extract_cells` par un
  appel à un petit CNN entraîné sur un dataset public de pièces
  d'échecs (ex. Chess Pieces Dataset sur Roboflow), classant chacune des
  64 cases parmi 13 états (vide + 6 pièces blanches + 6 pièces noires).
- **Un seul échiquier à la fois** : `main.py` garde un seul `BoardTracker`
  global. Pour plusieurs parties en parallèle, remplacer par un
  dictionnaire `{game_id: BoardTracker}` et ajouter l'identifiant de
  partie dans l'URL/les messages WebSocket.

## Structure du projet

```
chess-broadcast/
├── client/
│   └── index.html      # page unique : mode caméra + mode spectateur
├── server/
│   ├── main.py          # FastAPI, WebSockets, routes PGN/FEN
│   ├── vision.py         # calibration, découpage en cases, détection de coup
│   └── requirements.txt
└── README.md
```
