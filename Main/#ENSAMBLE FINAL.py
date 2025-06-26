#ENSAMBLE FINAL
import cv2
import numpy as np
import traceback
from pathlib import Path
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import shutil
import chess
import subprocess
import time
import socket
import time
import numpy as np
import time
import struct
import math
import os

# -------------------------
# CONFIGURACIÓN DEL ROBOT
# -------------------------

ip = ("192.168.0.100")  # Reemplaza con la IP de tu UR

# Posiciones físicas conocidas (en mm) de dos esquinas
p_h8 = np.array([-407.39, 21.30])   # Casilla H8 (origen local del tablero)
p_a8 = np.array([-197.15, 422.2])  # Casilla A8 (opuesta)

# Alturas y orientación del TCP
z = -45
z_agarre = z + 29
z_sobre = z + 160
rx, ry, rz = 3.08, 0.79, 0.095

# -------------------------
# MATRIZ DE TRANSFORMACIÓN
# -------------------------

# Ángulo de rotación en Z (en radianes)
theta = math.atan2(p_a8[1] - p_h8[1], p_a8[0] - p_h8[0])
print ("Ángulo de rotación (radianes):", theta)

deg = math.degrees(theta)
print("Ángulo de rotación (grados):", deg)

# Traslación
x, y = p_h8[0], p_h8[1]

# Matriz de transformación
T = np.array([
    [np.cos(theta), -np.sin(theta), 0, x],
    [np.sin(theta),  np.cos(theta), 0, y],
    [0,              0,             1, z],
    [0,              0,             0, 1]
])

def transform_point(X, Y):
    # Punto en el sistema local
    p_local = np.array([X, Y, 0, 1])

    # Punto transformado al sistema global
    p_global = T @ p_local
    return p_global[:2]  # Retornar solo x, y

# -------------------------
# GENERACIÓN DE CASILLAS CORREGIDA
# -------------------------

casillas_agarre = {}
casillas_sobre = {}
columnas = "hgfedcba"
filas = "87654321"  # Filas en orden inverso (8=arriba, 1=abajo)

for col in range(8):     # columnas (a=0, h=7)
    for fila in range(8): # filas (8=0, 1=7)
        nombre = f"{columnas[col]}{filas[fila]}"
        
        # Posición local en sistema de tablero
        local_x = ((col * 56) + 28)
        local_y = ((fila * 56) + 39)

        
        pos_global = transform_point(local_x, local_y)
        
        # Almacenar posiciones
        casillas_agarre[nombre] = [*pos_global[:2], z_agarre, rx, ry, rz]
        casillas_sobre[nombre] = [*pos_global[:2], z_sobre, rx, ry, rz]

# -------------------------
# POSICIÓN HOME
# -------------------------

# Calcular posición segura frente al tablero

home = pos_global = transform_point(((4 * 56) + 28), -60)
z_home = z + 300
pose_home = [home[0], home[1], z_home, rx, ry, rz]


# -------------------------
# CONTROL DE LA PINZA ROBOTIQ
# -------------------------

def send_command(cmd):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip, 63352))
        s.sendall((cmd + '\n').encode())
        data = s.recv(1024)
    return data.decode().strip()


def gripper_open():
    print("Abriendo gripper...")
    send_command("SET POS 100")
    time.sleep(1)

def gripper_close():
    print("Cerrando gripper...")
    send_command("SET POS 250")
    time.sleep(1)

def gripper_activate():
    #activar gripper
    print("Activando gripper...")
    send_command("SET ACT 1")
    time.sleep(1)

    # Establecer velocidad y fuerza
    send_command("SET SPE 150")  # velocidad (0–255)
    send_command("SET FOR 250")  # fuerza (0–255)
    time.sleep(0.5)


# -------------------------
# FUNCIONES DE MOVIMIENTO
# -------------------------

def get_actual_tcp_pose(ip):
    """
    Obtiene la posición actual del TCP en mm y radianes desde el UR a través del puerto 30003.
    Devuelve: [x_mm, y_mm, z_mm, rx, ry, rz]
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((ip, 30003))

            # Leer el primer paquete completo (1108 bytes)
            data = b''
            while len(data) < 1108:
                more = s.recv(1108 - len(data))
                if not more:
                    raise Exception("Conexión cerrada prematuramente.")
                data += more

            # Los valores de la pose están en los bytes 444 a 491
            pose_bytes = data[444:492]  # 6 * 8 bytes = 48 bytes
            pose = struct.unpack('!6d', pose_bytes)

            # Convertir posición a mm
            x_mm = pose[0] * 1000
            y_mm = pose[1] * 1000
            z_mm = pose[2] * 1000
            rx, ry, rz = pose[3:]

            return [x_mm, y_mm, z_mm, rx, ry, rz]

    except Exception as e:
        print(f"[ERROR] No se pudo leer la posición actual del TCP: {e}")
        return None
    
def move_to(pose_mm_rad, a=1.2, v=0.5, wait=True, position_tolerance=0.5, orientation_tolerance=0.01):
    """
    Envía un movimiento con movej a una pose, y espera si wait=True.
    pose_mm_rad: [x_mm, y_mm, z_mm, rx, ry, rz]
    a, v: aceleración y velocidad
    position_tolerance: en mm
    orientation_tolerance: en radianes
    """
    x, y, z = [p / 1000.0 for p in pose_mm_rad[:3]]
    rx, ry, rz = pose_mm_rad[3:]

    # Generar script UR
    script = f"""
def move_to_pose():
    target_pose = p[{x}, {y}, {z}, {rx}, {ry}, {rz}]
    target_q = get_inverse_kin(target_pose)
    movej(target_q, a={a}, v={v})
end
move_to_pose()
"""

    print("Enviando URScript:\n", script)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, 30002))
    sock.send(script.encode('utf-8'))
    time.sleep(0.5)
    sock.close()

    if not wait:
        return

    # Esperar hasta llegar a destino
    target = np.array([p / 1000.0 if i < 3 else p for i, p in enumerate(pose_mm_rad)])
    pos_tol = position_tolerance / 1000.0

    print("Esperando a que el robot llegue al destino...")

    max_wait_time = 15  # segundos
    start_time = time.time()

    while True:
        current = get_actual_tcp_pose(ip)
        if current is None:
            time.sleep(0.1)
            continue

        current = np.array([p / 1000.0 if i < 3 else p for i, p in enumerate(current)])
        pos_error = np.linalg.norm(current[:3] - target[:3])
        orient_error = np.linalg.norm(current[3:] - target[3:])
        #print(f"Posición actual: {current[:3]}, Error de posición: {pos_error:.3f} mm")
        #print(f"Orientación actual: {current[3:]}, Error de orientación: {orient_error:.3f} rad")

        if pos_error < pos_tol:
            print("Robot llegó a destino.")
            break

        if time.time() - start_time > max_wait_time:
            print("[ADVERTENCIA] Tiempo de espera superado.")
            break

        """"
        if pos_error < pos_tol and orient_error < orientation_tolerance:
            print("Robot llegó a destino.")
            break

        if time.time() - start_time > max_wait_time:
            print("[ADVERTENCIA] Tiempo de espera superado.")
            break
        """

        time.sleep(0.05)





def pick(casilla):
    move_to(pose_mm_rad=casillas_sobre[casilla], a=1.2, v=0.5) 
    move_to(pose_mm_rad=casillas_agarre[casilla], a=1.2, v=0.5)
    gripper_close()
    move_to(pose_mm_rad=casillas_sobre[casilla], a=1.2, v=0.5)

def place(casilla):
    move_to(pose_mm_rad=casillas_sobre[casilla], a=1.2, v=0.5) 
    move_to(pose_mm_rad=casillas_agarre[casilla], a=1.2, v=0.5)
    gripper_open()
    move_to(pose_mm_rad=casillas_sobre[casilla], a=1.2, v=0.5)

def go_home():
    print("Volviendo a home...")
    move_to(pose_mm_rad=pose_home, a=1.2, v=0.5)

def picture_to_home():
    move_to(pose_mm_rad=[-495, 120, 737, 1.474, -0.455, -0.302], a=1.2, v=0.5)
    move_to(pose_mm_rad=pose_home, a=1.2, v=0.5)

def take_picture():
    move_to(pose_mm_rad=[-388, 10, 790, 1.546, 0.076, -0.585], a=1.2, v=0.5)
    move_to(pose_mm_rad=[-390, 328, 961, 0.108, 0.302, 5.867], a=1.2, v=0.5)

def to_end():
    move_to(pose_mm_rad=[-388, 10, 790, 1.546, 0.076, -0.585], a=1.2, v=0.5)
    end = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    end.connect((ip, 30002))

    # Mover el robot
    script = "movej([3.1416, -1.5708, 0.0, 0.0, 0.0, 0.0], a=1.2, v=0.5)\n"
    end.send(script.encode('utf-8'))
    time.sleep(0.5)
    end.close()

def move_pieza(casilla_origen, casilla_destino):
    """
    Mueve una pieza desde la casilla de origen a la casilla de destino.
    """
    pick(casilla_origen)
    place(casilla_destino)


# ────────────────────────────────────────────────────────────────
# 1. DETECTOR ROBUSTO DE ESQUINAS 7×7 (CLAHE + multi-escala/rot)
# ────────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────────
# 3. PROCESA UN TABLERO: homografía única para la malla 9×9
# ────────────────────────────────────────────────────────────────
def procesar_tablero_chess(ruta_imagen, mostrar=True):
    base   = Path("C:/Users/Alegu/OneDrive/Documentos/ROBOTICA/UR5 IMAGE DETECTION")
    nombre = Path(ruta_imagen).stem
    out_v  = base / "casillas" / nombre / "vacias"
    out_o  = base / "casillas" / nombre / "ocupadas"
    out_v.mkdir(parents=True, exist_ok=True)
    out_o.mkdir(exist_ok=True)

    img = cv2.imread(str(ruta_imagen))
    if img is None:
        print("No se pudo abrir:", ruta_imagen)
        return False, None, None # Return False, None for corners, None for image

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img, (720, 720))

    ok, corners = detectar_esquinas_sb(img)
    if not ok:
        print("No se encontraron esquinas en", nombre)
        return False, None, None # Return False, None for corners, None for image

    # A) puntos ideales 7×7 internos (1..7 en x, 1..7 en y)
    obj7 = np.array([[x+1, y+1] for y in range(7) for x in range(7)],
                    np.float32)

    # B) homografía global
    H, _ = cv2.findHomography(obj7, corners.reshape(-1, 2))

    # C) malla 9×9 ideal (0..8) → proyectada
    obj9 = np.array([[x, y] for y in range(9) for x in range(9)],
                    np.float32).reshape(-1, 1, 2)
    g9 = cv2.perspectiveTransform(obj9, H).reshape(9, 9, 2)

    # (opcional) afinado sub-píxel
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.cornerSubPix(gray, g9.reshape(-1, 1, 2), (5, 5), (-1, -1), crit)

    # D) recorte de 64 casillas
    res = img.copy()
    for r in range(8):
        for c in range(8):
            pts = np.array([g9[r, c], g9[r, c+1],
                            g9[r+1, c+1], g9[r+1, c]], np.float32)
            dst = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
            M   = cv2.getPerspectiveTransform(pts, dst)
            cell= cv2.warpPerspective(img, M, (100, 100))

            blanca = (r + c) % 2 == 0
            pieza  = detectar_pieza_en_casilla(cell, blanca)
            name   = f"{chr(97+c)}{8-r}.jpg"
            cv2.imwrite(str((out_o if pieza else out_v) / name), cell)

            col = (0, 0, 255) if pieza else (0, 255, 0)
            cv2.polylines(res, [pts.astype(int)], True, col, 2)

    if mostrar:
        cv2.imshow(f"Resultado: {nombre}", res)
    print("Procesado:", nombre)
    return True, corners, res # Return True, corners, and the image with detected corners

TRADUCCION_A_FEN = {
    "ReyNegro": "k", "DamaNegra": "q", "TorreNegra": "r", "AlfilNegro": "b", "CaballoNegro": "n", "peonNegro": "p",
    "ReyBlanco": "K", "DamaBlanca": "Q", "TorreBlanca": "R", "AlfilBlanco": "B", "CaballoBlanco": "N", "peonBlanco": "P",
    "Vacia": "empty"
}

# ————————————————————————————————————————————————————————————————
# BLOQUE 2: Clasificación CNN + armado de matriz (pasos 9–12)
# ————————————————————————————————————————————————————————————————

def clasificar_y_actualizar_tablero(
    ruta_imagen,
    modelo_h5=r"C:\Users\Alegu\OneDrive\Documentos\ROBOTICA\UR5 IMAGE DETECTION\RedesGPU\best_ajedrez1.keras",
    data_dir=r"C:\Users\Alegu\OneDrive\Documentos\ROBOTICA\UR5 IMAGE DETECTION\Data_Set_Definitivo\Data_Set_Sintetizado2"
):
    # procesar_tablero_chess now returns (ok, corners, img_with_corners_for_display)
    ok_processed, _, img_rot_display = procesar_tablero_chess(ruta_imagen)

    if not ok_processed:
        # If processing failed, return empty results and the original img (or None)
        return [], [["empty"] * 8 for _ in range(8)], img_rot_display

    base = Path("C:/Users/Alegu/OneDrive/Documentos/ROBOTICA/UR5 IMAGE DETECTION")
    nombre = Path(ruta_imagen).stem
    carpeta_ocupadas = base / "casillas" / nombre / "ocupadas"

    model = load_model(modelo_h5)
    clases = sorted(d.name for d in Path(data_dir).iterdir() if d.is_dir())

    resultados = []
    board = [["empty"] * 8 for _ in range(8)]
    for img_path in carpeta_ocupadas.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (90,90)).astype("float32") / 255.0
        img = np.expand_dims(img, 0)
        preds = model.predict(img, verbose=0)[0]

        pieza_es = clases[np.argmax(preds)]
        pieza_en = TRADUCCION_A_FEN.get(pieza_es, "empty")

        resultados.append((img_path.name, pieza_es))
        col = ord(img_path.stem[0]) - 97
        fila = 8 - int(img_path.stem[1])
        board[fila][col] = pieza_en

    if carpeta_ocupadas.exists():
        shutil.rmtree(carpeta_ocupadas.parent, ignore_errors=True) # Remove the parent folder to clean both occupied and empty

    return resultados, board, img_rot_display

# =================================================================
# DICCIONARIO UNICODE PARA DIBUJAR
# =================================================================

UNICODE_PIECES = {
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
    "K": "♚", "Q": "♛","R": "♜", "B": "♝", "N": "♞", "P": "♟",
    "empty": "·"
}

def dibujar_tablero_unicode(board_matrix):
    fig, ax = plt.subplots(figsize=(7,7))
    for rank in range(8):
        for file in range(8):
            color = "#f0d9b5" if (rank + file) % 2 == 0 else "#b58863"
            ax.add_patch(plt.Rectangle((file, 7 - rank), 1, 1, facecolor=color, edgecolor="none"))
            pieza = board_matrix[rank][file]
            if pieza != "empty":
                text_color = "white" if pieza.isupper() else "black"
                ax.text(file + 0.5, 7 - rank + 0.5, UNICODE_PIECES[pieza],
                        fontsize=32, ha="center", va="center", color=text_color)
            else:
                ax.text(file + 0.5, 7 - rank + 0.5, UNICODE_PIECES["empty"],
                        fontsize=32, ha="center", va="center", color="grey")

    for rank_num in range(8):
        ax.text(-0.5, rank_num + 0.5, str((rank_num+1)), # Retained original rank numbering
                fontsize=14, ha="center", va="center", color="black")
    for file_char_code in range(8):
        ax.text(file_char_code + 0.5, -0.5, chr(97 + file_char_code),
                fontsize=14, ha="center", va="center", color="black")

    ax.set_xlim(-1, 8.5)
    ax.set_ylim(-1, 8.5)
    ax.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

def tablero_a_FEN(board_matrix):
    board_fen = ""
    for row in board_matrix:
        empty_count = 0
        for piece_char in row:
            if piece_char == "empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    board_fen += str(empty_count)
                    empty_count = 0
                board_fen += piece_char
        if empty_count > 0:
            board_fen += str(empty_count)
        board_fen += "/"
    return board_fen.rstrip("/")

# ————————————————————————————————————————————————————————————————
# HANDSHAKE COMPLETO UCI PARA STOCKFISH (SÍNCRONO)
# ————————————————————————————————————————————————————————————————

def get_stockfish_move_sync(fen_string, stockfish_path, difficulty_level, time_limit=0.1):
    """
    Implementación síncrona UCI con handshake completo.
    """
    proc = None
    try:
        proc = subprocess.Popen(
            stockfish_path,
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        def send(cmd):
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()

        def read_until(token):
            while True:
                line = proc.stdout.readline().strip()
                if line == token:
                    return

        # 1) Inicialización UCI
        send("uci")
        read_until("uciok")

        # 2) Configurar nivel
        send(f"setoption name Skill Level value {difficulty_level}")

        # 3) Asegurar que el motor está listo
        send("isready")
        read_until("readyok")

        # 4) Fijar posición
        send(f"position fen {fen_string}")

        # 5) Otra vez asegurar
        send("isready")
        read_until("readyok")

        # 6) Lanzar búsqueda
        send(f"go movetime {int(time_limit * 1000)}")

        # 7) Leer hasta bestmove
        best_move = None
        while True:
            line = proc.stdout.readline().strip()
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "(none)":
                    best_move = chess.Move.from_uci(parts[1])
                break

        return best_move

    except Exception as e:
        print(f"Error Stockfish: {e}")
        traceback.print_exc()
        return None

    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=1.0)

def tomar_foto(nombre_archivo="foto.jpg", ruta_guardado="./"):
    # Asegúrate de que la ruta existe
    os.makedirs(ruta_guardado, exist_ok=True)

    # Inicializa la cámara (0 es la cámara por defecto)
    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    # Espera a que la cámara se estabilice (opcional)
    cv2.waitKey(500)

    # Captura una sola imagen
    ret, imagen = camara.read()

    if ret:
        ruta_completa = os.path.join(ruta_guardado, nombre_archivo)
        cv2.imwrite(ruta_completa, imagen)
        print(f"Foto guardada en: {ruta_completa}")
    else:
        print("Error: No se pudo capturar la imagen.")

    # Libera la cámara
    camara.release()
    cv2.destroyAllWindows()

# ————————————————————————————————————————————————————————————————
# FUNCIÓN PRINCIPAL SÍNCRONA
# ————————————————————————————————————————————————————————————————

def main_sync():
    # --- VARIABLES DE CONFIGURACIÓN PARA EL USUARIO ---
    STOCKFISH_PATH = r"C:\Users\Alegu\OneDrive\Documentos\ROBOTICA\UR5 IMAGE DETECTION\stockfish-windows-x86-64-avx2.exe"
    USER_COLOR      = "black"   # "white" o "black"
    DIFFICULTY_LEVEL = 20       # Nivel de dificultad (0-20)
    # ——————————————————————————————————————————————————
    gripper_open()
    take_picture()
    ruta_imagen_tablero = r"C:\Users\Alegu\OneDrive\Documentos\ROBOTICA\UR5 IMAGE DETECTION\tableros\Camera Roll\Camera Roll"
    tomar_foto("jugada-actual.jpg", ruta_imagen_tablero)
    picture_to_home()

    try:
        ruta_imagen_tablero = r"C:\Users\Alegu\OneDrive\Documentos\ROBOTICA\UR5 IMAGE DETECTION\tableros\Camera Roll\Camera Roll\jugada-actual.jpg"
        resultados, board_matrix, img_rot_display = clasificar_y_actualizar_tablero(ruta_imagen_tablero)


        if any(p != "empty" for row in board_matrix for p in row):
            if img_rot_display is not None:
                img = cv2.cvtColor(img_rot_display, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(12,12))
                plt.imshow(img)
                plt.title("Tablero Real (Rotado 180°)", fontsize=20)
                plt.axis("off")
                plt.tight_layout()
                plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
                plt.show()

            fen_piezas = tablero_a_FEN(board_matrix)
            # Turno corregido:
            fen_turn = 'w' if USER_COLOR == 'white' else 'b'
            fen_completo = f"{fen_piezas} {fen_turn} - - 0 1"
            print(f"\n--- FEN Generado ---\n{fen_completo}\n")

            board_obj = chess.Board(fen_completo)
            print("Tablero con python-chess:")
            print(board_obj)

            print(f"\nStockfish (dificultad {DIFFICULTY_LEVEL}) está pensando...")
            best_move = get_stockfish_move_sync(fen_completo, STOCKFISH_PATH, DIFFICULTY_LEVEL)

            if best_move:
                print(f"Mejor jugada de Stockfish: {best_move}")
                board_obj.push(best_move)
                print("\nTablero después de la jugada sugerida:")
                print(board_obj)
                casillainicial = chess.square_name(best_move.from_square)
                casillafinal = chess.square_name(best_move.to_square)
                print(f'casilla inicial: {casillainicial}')
                print(f'casillafinal:{casillafinal}')
                move_pieza(casillainicial, casillafinal)
                go_home()

            else:
                print("Stockfish no pudo encontrar una jugada o la partida terminó.")
        else:
            print("No se detectaron piezas, se omite FEN y Stockfish.")

        dibujar_tablero_unicode(board_matrix)
        cv2.waitKey(0) # Added to keep the OpenCV window open
        cv2.destroyAllWindows() # Added to close all OpenCV windows

    except FileNotFoundError as fe:
        print(f"Error de archivo: {fe}. Verifica rutas.")
    except Exception as e:
        print("Error general en la integración:", e)
        traceback.print_exc()
    
    

# ————————————————————————————————————————————————————————————————
# EJECUCIÓN PRINCIPAL
# ————————————————————————————————————————————————————————————————

if __name__ == "__main__":
    main_sync()
