# CNN.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent

# Configuración de rutas

modelo_path = BASE_DIR / "Modelos" / "best_ajedrez1.keras"
dataset_path = BASE_DIR / "Data_set" / "Data_Set_Sintetizado2"
imagen_path = BASE_DIR / "Main" / "jugada_actual.png"


TRADUCCION_A_FEN = {
    "ReyNegro": "k", "DamaNegra": "q", "TorreNegra": "r", "AlfilNegro": "b", "CaballoNegro": "n", "peonNegro": "p",
    "ReyBlanco": "K", "DamaBlanca": "Q", "TorreBlanca": "R", "AlfilBlanco": "B", "CaballoBlanco": "N", "peonBlanco": "P",
    "Vacia": "empty"
}

class VisionEngine:
    def __init__(self, base_output="casillas"):
        self.image_path   = Path(imagen_path)
        self.model = load_model(modelo_path)
        self.dataset_path = Path(dataset_path)
        self.base_output = Path(base_output)
        self.clases = sorted(d.name for d in self.dataset_path.iterdir() if d.is_dir())

    # ────────────────────────────────────────────────────────────────
    # 1. DETECTOR ROBUSTO DE ESQUINAS 7×7 (CLAHE + multi-escala/rot)
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def detectar_esquinas_sb(img_bgr, pattern=(7, 7)):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
        gray = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        escalas     = [1.0, 0.75, 0.5]
        rotaciones = [0, 90, 180, 270]
        rot_flag    = {0: None,
                        90:  cv2.ROTATE_90_CLOCKWISE,
                        180: cv2.ROTATE_180,
                        270: cv2.ROTATE_90_COUNTERCLOCKWISE}

        flags = (cv2.CALIB_CB_EXHAUSTIVE |
                cv2.CALIB_CB_ACCURACY   |
                cv2.CALIB_CB_NORMALIZE_IMAGE)

        H, W = gray.shape
        for s in escalas:
            g_s = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            for r in rotaciones:
                test = cv2.rotate(g_s, rot_flag[r]) if rot_flag[r] else g_s
                ok, c = cv2.findChessboardCornersSB(test, pattern, flags)
                if ok:
                    c = c.astype(np.float32) / s
                    if r:
                        M = cv2.getRotationMatrix2D((W/2, H/2), -r, 1.0)
                        c_h = np.hstack([c.squeeze(), np.ones((49, 1))])
                        c   = (M @ c_h.T).T.reshape(-1, 1, 2)
                    return True, np.ascontiguousarray(c, np.float32)
        return False, None


    # ────────────────────────────────────────────────────────────────
    # 2. DETECCIÓN DE PIEZA EN UNA CASILLA
    # ────────────────────────────────────────────────────────────────
    
    @staticmethod
    def detectar_pieza_en_casilla(casilla, es_blanca, margen_rel=0.20):
        h, w = casilla.shape[:2]
        m    = int(min(h, w) * margen_rel)
        roi  = casilla[m:h-m, m:w-m]

        g   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dif = cv2.GaussianBlur(g, (5, 5), 0)
        bin_inv = cv2.adaptiveThreshold(dif, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 15, 3)
        k = np.ones((3, 3), np.uint8)
        bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN,   k, 1)
        bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, k, 2)

        cnts, _   = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_rel  = 0
        if cnts:
            c_max = max(cnts, key=cv2.contourArea)
            area_rel = cv2.contourArea(c_max) / bin_inv.size
            (cx, cy), _ = cv2.minEnclosingCircle(c_max)
            if not (0.25*w < cx+m < 0.75*w and 0.25*h < cy+m < 0.75*h):
                area_rel = 0

        edge_density = np.count_nonzero(cv2.Canny(dif, 40, 120)) / dif.size
        var_tex      = np.var(g)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v   = hsv[..., 2]
        dark_ratio = np.count_nonzero(v < np.median(v)*0.80) / v.size

        if es_blanca:
            return (area_rel > 0.03) or (edge_density > 0.045) or (var_tex > 180)
        else:
            return (area_rel > 0.055) or (edge_density > 0.055) or \
                (var_tex > 210)  or (dark_ratio > 0.12)



    def procesar_tablero_chess(self, mostrar=True):
        ruta_imagen = self.image_path
        nombre = ruta_imagen.stem
        out_v  = self.base_output / nombre / "vacias"
        out_o  = self.base_output / nombre / "ocupadas"
        out_v.mkdir(parents=True, exist_ok=True)
        out_o.mkdir(exist_ok=True)

        img = cv2.imread(str(ruta_imagen))
        if img is None:
            print("No se pudo abrir:", ruta_imagen)
            return False, None, None

        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.resize(img, (720, 720))

        ok, corners = self.detectar_esquinas_sb(img)
        if not ok:
            print("No se encontraron esquinas en", nombre)
            return False, None, None

        obj7 = np.array([[x+1, y+1] for y in range(7) for x in range(7)], np.float32)
        H, _ = cv2.findHomography(obj7, corners.reshape(-1, 2))

        obj9 = np.array([[x, y] for y in range(9) for x in range(9)],
                        np.float32).reshape(-1, 1, 2)
        g9 = cv2.perspectiveTransform(obj9, H).reshape(9, 9, 2)

        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.cornerSubPix(gray, g9.reshape(-1, 1, 2), (5, 5), (-1, -1), crit)

        res = img.copy()
        for r in range(8):
            for c in range(8):
                pts = np.array([g9[r, c], g9[r, c+1], g9[r+1, c+1], g9[r+1, c]], np.float32)
                dst = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
                M   = cv2.getPerspectiveTransform(pts, dst)
                cell = cv2.warpPerspective(img, M, (100, 100))

                blanca = (r + c) % 2 == 0
                pieza  = self.detectar_pieza_en_casilla(cell, blanca)
                name   = f"{chr(97+c)}{8-r}.jpg"
                cv2.imwrite(str((out_o if pieza else out_v) / name), cell)

                col = (0, 0, 255) if pieza else (0, 255, 0)
                cv2.polylines(res, [pts.astype(int)], True, col, 2)

        if mostrar:
            cv2.imshow(f"Resultado: {nombre}", res)
        print("Procesado:", nombre)
        return True, corners, res


    def clasificar_casillas(self, nombre_img):
        carpeta_ocupadas = self.base_output / nombre_img / "ocupadas"
        resultados = []
        board = [["empty"] * 8 for _ in range(8)]

        for img_path in carpeta_ocupadas.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (90,90)).astype("float32") / 255.0
            img = np.expand_dims(img, 0)
            preds = self.model.predict(img, verbose=0)[0]

            pieza_es = self.clases[np.argmax(preds)]
            pieza_en = TRADUCCION_A_FEN.get(pieza_es, "empty")
            resultados.append((img_path.name, pieza_es))

            col = ord(img_path.stem[0]) - 97
            fila = 8 - int(img_path.stem[1])
            board[fila][col] = pieza_en

        # Limpieza
        if carpeta_ocupadas.exists():
            shutil.rmtree(carpeta_ocupadas.parent, ignore_errors=True)

        return resultados, board

    def board_a_fen(self, board_matrix):
        fen = ""
        for row in board_matrix:
            empty = 0
            for piece in row:
                if piece == "empty":
                    empty += 1
                else:
                    if empty:
                        fen += str(empty)
                        empty = 0
                    fen += piece
            if empty:
                fen += str(empty)
            fen += "/"
        return fen.rstrip("/")

    def procesar_imagen(self, mostrar=True):
        nombre_img = self.image_path.stem  

        ok, _, img_rot_display = self.procesar_tablero_chess(mostrar)
        if not ok:
            return None, [["empty"]*8 for _ in range(8)], img_rot_display

        resultados, board = self.clasificar_casillas(nombre_img)
        fen = self.board_a_fen(board)
        return fen, board, img_rot_display

"""
# Ejemplo de uso:
# from CNN import VisionEngine

engine = VisionEngine()

fen, board, imagen_procesada = engine.procesar_imagen(mostrar=False)

print("FEN generado:", fen)
for fila in board:
    print(fila)
"""