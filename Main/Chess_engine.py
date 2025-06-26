# Chess_engine.py

import subprocess
import chess
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Configuración de rutas

stockfish_path = BASE_DIR / "Stockfish" / "stockfish-windows-x86-64-avx2.exe"  # path a Stockfish

class ChessEngine:
    def __init__(self, player_color="white", player_elo=1500, stockfish_elo=2200):
        self.proc = subprocess.Popen(
            stockfish_path,
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1
        )
        self.color = player_color.lower()
        self.player_elo = player_elo
        self.stockfish_elo = stockfish_elo
        self.board = chess.Board()  # Historial interno

        self._init_engine()

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token):
        while True:
            line = self.proc.stdout.readline().strip()
            if line == token:
                return

    def _init_engine(self):
        self._send("uci")
        self._read_until("uciok")
        self._send(f"setoption name UCI_LimitStrength value true")
        self._send(f"setoption name UCI_Elo value {self.stockfish_elo}")
        self._send("isready")
        self._read_until("readyok")

    def reset(self):
        """Reinicia el tablero y el historial."""
        self.board.reset()

    def get_best_move(self, fen):
        """
        Actualiza el estado con el FEN, calcula mejor jugada y detecta:
            - si es captura
            - si es enroque
            - si la partida ha terminado y cómo
        """
        self.board = chess.Board(fen)
        moves_str = " ".join(move.uci() for move in self.board.move_stack)
        self._send(f"position fen {fen}")
        self._send("go movetime 200")

        best_move = None
        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith("bestmove"):
                move_str = line.split()[1]
                if move_str == "(none)":
                    return None, False, False, self._get_game_over_status()
                best_move = chess.Move.from_uci(move_str)
                break

        is_capture = self.board.is_capture(best_move)
        is_castling = self.board.is_castling(best_move)

        self.board.push(best_move)

        return best_move, is_capture, is_castling, self._get_game_over_status()

    def get_hint(self):
        """Devuelve una pista para el jugador humano (usando el color opuesto al del motor)."""
        self._send(f"setoption name UCI_Elo value {self.player_elo}")
        moves_str = " ".join(move.uci() for move in self.board.move_stack)
        self._send(f"position startpos moves {moves_str}")
        self._send("go movetime 200")

        best_move = None
        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith("bestmove"):
                move_str = line.split()[1]
                if move_str == "(none)":
                    return None
                best_move = chess.Move.from_uci(move_str)
                break

        self._send(f"setoption name UCI_Elo value {self.stockfish_elo}")  # Restaurar
        return best_move

    def get_move_history(self):
        return [move.uci() for move in self.board.move_stack]
    

    def _get_game_over_status(self):
        """
        Retorna el estado del juego:
            - 'player_mates_engine'
            - 'engine_mates_player'
            - 'draw'
            - None (si la partida sigue)
        """
        if self.board.is_checkmate():
            # El que no tiene el turno fue quien dio jaque mate
            last_player = not self.board.turn
            if (last_player and self.color == "white") or (not last_player and self.color == "black"):
                return "player_mates_engine"
            else:
                return "engine_mates_player"
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_fifty_moves() or self.board.can_claim_threefold_repetition():
            return "draw"
        else:
            return None


    def close(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=1.0)


"""
ejemplo de uso:

from Chess_engine import ChessEngine
import chess

engine = ChessEngine(player_color="white", player_elo=1400, stockfish_elo=2200)

fen_actual = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
best_move, is_capture, is_castling, game_status = engine.get_best_move(fen_actual)

if best_move:
    print("Mejor jugada:", best_move)
    print("Captura:", is_capture, "| Enroque:", is_castling)
else:
    print("No hay jugada posible (bestmove none)")

if game_status:
    print("Estado del juego:", game_status)


pista = engine.get_hint()
print("Pista para jugador humano:", pista)

print("Historial:", engine.get_move_history())

engine.close()
"""