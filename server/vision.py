"""
vision.py — Suivi d'un échiquier physique par webcam.

Principe (volontairement simple, sans réseau de neurones, pour être
utilisable immédiatement) :

1. L'opérateur clique les 4 coins du plateau dans l'image -> on calcule
   une homographie qui redresse l'échiquier en une image carrée 8x8.
2. On découpe cette image en 64 cases.
3. Au lieu de "reconnaître" chaque pièce, on compare l'image de chaque
   case entre deux instants stables : les cases qui changent = les cases
   "touchées" par le coup joué (départ, arrivée, tour qui roque, etc.)
4. On demande à python-chess la liste des coups légaux dans la position
   courante, et on choisit celui dont l'ensemble des cases attendu colle
   le mieux à l'ensemble des cases réellement touchées.

Limites connues (MVP) :
- Ne "voit" pas quelle pièce est jouée, seulement quelles cases bougent
  -> en cas d'ambiguïté rarissime, ou de promotion, il peut se tromper
  (la promotion est supposée en Dame par défaut).
- Sensible à l'éclairage/à l'ombre de la main : on attend une stabilité
  de plusieurs frames avant de considérer qu'un coup est terminé.
- Peut être remplacé plus tard par un classifieur CNN par case sans
  changer l'architecture globale (juste extract_cells + une fonction de
  classification en plus).
"""

import time
import numpy as np
import cv2
import chess
import chess.pgn


GRID_SIZE = 400          # taille (px) de l'image redressée
CELL = GRID_SIZE // 8     # taille d'une case redressée
CROP_MARGIN = 10          # on ignore les bords de case (lignes du plateau)

MOTION_THRESHOLD = 14.0   # frame->frame : en dessous = "rien ne bouge" (assoupli pour caméras de téléphone)
STABLE_FRAMES = 3         # nb de frames stables avant de figer un état
CHANGE_THRESHOLD = 20.0   # état->état : au dessus = "case modifiée"
MATCH_TOLERANCE = 1       # nb de cases de différence tolérées avec le coup attendu


def squares_touched_by(board: chess.Board, move: chess.Move) -> set:
    """Renvoie l'ensemble des cases (indices python-chess) censées changer
    visuellement pour ce coup : départ, arrivée, + cas spéciaux."""
    squares = {move.from_square, move.to_square}

    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if board.is_kingside_castling(move):
            squares |= {chess.square(7, rank), chess.square(5, rank)}
        else:
            squares |= {chess.square(0, rank), chess.square(3, rank)}

    if board.is_en_passant(move):
        captured = chess.square(chess.square_file(move.to_square),
                                 chess.square_rank(move.from_square))
        squares.add(captured)

    return squares


class BoardTracker:
    def __init__(self):
        self.M = None                 # matrice d'homographie
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.game.headers["Event"] = "Partie diffusée en direct"
        self.node = self.game

        self.confirmed_cells = None   # état "figé" de référence (64 vignettes)
        self.pending_cells = None     # dernière frame reçue, en attente de stabilité
        self.stable_count = 0
        self.last_move_time = time.time()

    # ---------- calibration ----------

    def set_corners(self, corners_xy, frame_w, frame_h):
        """corners_xy: liste de 4 points [x,y] en pixels de la frame envoyée
        par le client, dans l'ordre : haut-gauche(a8), haut-droit(h8),
        bas-droit(h1), bas-gauche(a1) — vu depuis la caméra, les Blancs
        étant du côté le plus proche de l'opérateur."""
        src = np.float32(corners_xy)
        dst = np.float32([[0, 0], [GRID_SIZE, 0], [GRID_SIZE, GRID_SIZE], [0, GRID_SIZE]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        # on repart d'un état vierge : le prochain frame stable devient la référence
        self.confirmed_cells = None
        self.pending_cells = None
        self.stable_count = 0

    def reset_game(self):
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.game.headers["Event"] = "Partie diffusée en direct"
        self.node = self.game
        self.confirmed_cells = None
        self.pending_cells = None
        self.stable_count = 0

    # ---------- traitement image ----------

    def _warp(self, frame_bgr):
        return cv2.warpPerspective(frame_bgr, self.M, (GRID_SIZE, GRID_SIZE))

    def _extract_cells(self, warped):
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        cells = [None] * 64
        for row in range(8):          # row 0 = rangée 8 (haut de l'image)
            for col in range(8):      # col 0 = colonne a (gauche de l'image)
                y0 = row * CELL + CROP_MARGIN
                y1 = (row + 1) * CELL - CROP_MARGIN
                x0 = col * CELL + CROP_MARGIN
                x1 = (col + 1) * CELL - CROP_MARGIN
                crop = gray[y0:y1, x0:x1]
                crop = cv2.resize(crop, (24, 24))
                square = chess.square(col, 7 - row)
                cells[square] = crop
        return cells

    @staticmethod
    def _cell_diff(a, b):
        return float(np.mean(cv2.absdiff(a, b)))

    def _match_move(self, touched_squares):
        best_move, best_score = None, None
        for move in self.board.legal_moves:
            expected = squares_touched_by(self.board, move)
            score = len(expected.symmetric_difference(touched_squares))
            if best_score is None or score < best_score:
                best_move, best_score = move, score
        if best_score is not None and best_score <= MATCH_TOLERANCE:
            return best_move
        return None

    def process_frame(self, frame_bgr):
        """À appeler à chaque frame reçue du client. Renvoie un dict
        décrivant ce qu'il s'est passé, ou None si rien de notable."""
        if self.M is None:
            return {"status": "no_calibration"}

        warped = self._warp(frame_bgr)
        cells = self._extract_cells(warped)

        if self.confirmed_cells is None:
            self.confirmed_cells = cells
            self.pending_cells = cells
            self.stable_count = 0
            return {"status": "baseline_set", "fen": self.board.fen()}

        if self.pending_cells is None:
            self.pending_cells = cells
            self.stable_count = 1
            return {"status": "tracking", "motion": 0.0, "stable": 1, "needed": STABLE_FRAMES}

        frame_motion = max(self._cell_diff(cells[s], self.pending_cells[s]) for s in range(64))
        self.pending_cells = cells

        if frame_motion > MOTION_THRESHOLD:
            self.stable_count = 0
            return {"status": "tracking", "motion": round(frame_motion, 1), "stable": 0, "needed": STABLE_FRAMES}

        self.stable_count += 1
        if self.stable_count < STABLE_FRAMES:
            return {"status": "tracking", "motion": round(frame_motion, 1), "stable": self.stable_count, "needed": STABLE_FRAMES}

        # Image stable depuis assez longtemps : comparer à la référence
        diffs = {s: self._cell_diff(cells[s], self.confirmed_cells[s]) for s in range(64)}
        touched = {s for s, d in diffs.items() if d > CHANGE_THRESHOLD}

        self.stable_count = 0
        if not touched:
            return {"status": "tracking", "motion": round(frame_motion, 1), "stable": 0, "needed": STABLE_FRAMES}

        move = self._match_move(touched)
        self.confirmed_cells = cells  # on fige le nouvel état dans tous les cas

        if move is None:
            return {"status": "uncertain", "touched": sorted(touched)}

        san = self.board.san(move)
        self.board.push(move)
        self.node = self.node.add_variation(move)
        self.last_move_time = time.time()

        return {
            "status": "move",
            "san": san,
            "fen": self.board.fen(),
            "pgn": str(self.game),
            "turn": "white" if self.board.turn else "black",
            "is_checkmate": self.board.is_checkmate(),
            "is_check": self.board.is_check(),
            "is_game_over": self.board.is_game_over(),
        }
