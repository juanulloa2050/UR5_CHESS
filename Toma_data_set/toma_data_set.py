#pathing con recorte para dataset
import cv2
import numpy as np
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk 
from PIL import Image, ImageTk
import random
import os
import re

# Configuración global de rutas
RUTA_BASE = Path("../Data_set")
RUTA_BASE_GUI = Path("../Imagenes_GUI")
RUTA_DATASET = RUTA_BASE / "Data_Set_Definitivo"

camara = 0  # Cambiar a 1 si se quiere usar la cámara

# Definiciones de piezas mediante listas de posiciones (notación algebraica)
peon_blanco    = []
peon_negro     = []
torre_blanca   = []
torre_negra    = []
caballo_blanco = []
caballo_negro  = []
alfil_blanco   = []
alfil_negro    = []
reina_blanca   = []
reina_negra    = []
rey_blanco     = []
rey_negro      = []

# Mapeo de piezas a carpetas
CARPETAS_PIEZAS = {
    'PB': "peonBlanco",
    'PN': "peonNegro",
    'TB': "TorreBlanca",
    'TN': "TorreNegra",
    'CB': "CaballoBlanco",
    'CN': "CaballoNegro",
    'AB': "AlfilBlanco",
    'AN': "AlfilNegro",
    'DB': "DamaBlanca",
    'DN': "DamaNegra",
    'RB': "ReyBlanco",
    'RN': "ReyNegro",
    '--': "Vacia"
}

# Matriz 8×8 que representa el estado del tablero en memoria
TABLERO = [['--' for _ in range(8)] for _ in range(8)]

# Contador global de cuántas imágenes de cada pieza llevamos guardadas
contador_piezas = {codigo: 0 for codigo in CARPETAS_PIEZAS.keys()}


def crear_estructura_carpetas():
    """Crea la estructura de carpetas si no existe"""
    for carpeta in CARPETAS_PIEZAS.values():
        (RUTA_DATASET / carpeta).mkdir(parents=True, exist_ok=True)


def convertir_posicion(posicion):
    """
    Convierte posición de ajedrez (ej: 'a1') a índices de matriz (fila, col) ∈ [0..7].
    'a1' → (7,0), 'a8' → (0,0), 'h1' → (7,7), 'h8' → (0,7).
    """
    col_char = posicion[0].lower()
    fila_num = int(posicion[1])
    fila = 8 - fila_num  # La fila 1 del ajedrez es la fila 7 en la matriz
    col = ord(col_char) - ord('a')  # 'a' -> 0, 'b' -> 1, etc.
    return fila, col


def configurar_tablero_desde_listas():
    """Configura el tablero desde las listas de posiciones"""
    # Configurar peones blancos
    for pos in peon_blanco:
        f, c = convertir_posicion(pos)
        TABLERO[f][c] = 'PB'
    # Configurar peones negros
    for pos in peon_negro:
        f, c = convertir_posicion(pos)
        TABLERO[f][c] = 'PN'
    # Configurar torres blancas
    for pos in torre_blanca:
        f, c = convertir_posicion(pos)
        TABLERO[f][c] = 'TB'
    # Configurar torres negras
    for pos in torre_negra:
        f, c = convertir_posicion(pos)
        TABLERO[f][c] = 'TN'
    # Configurar caballos blancos
    for pos in caballo_blanco:
        f, c = convertir_posicion(pos)
        TABLERO[f][c] = 'CB'
    # Configurar caballos negros
    for pos in caballo_negro:
        f, c = convertir_posicion(pos)
        TABLERO[f][c] = 'CN'
    # Configurar alfiles blancos
    for pos in alfil_blanco:
        f, c = convertir_posicion(pos)
        TABLERO[f][c] = 'AB'
    # Configurar alfiles negros
    for pos in alfil_negro:
        f, c = convertir_posicion(pos)
        TABLERO[f][c] = 'AN'
    # Configurar reinas blancas
    for pos in reina_blanca:
        f, c = convertir_posicion(pos)
        TABLERO[f][c] = 'DB'
    # Configurar reinas negras
    for pos in reina_negra:
        f, c = convertir_posicion(pos)
        TABLERO[f][c] = 'DN'
    # Configurar reyes blancos
    for pos in rey_blanco:
        f, c = convertir_posicion(pos)
        TABLERO[f][c] = 'RB'
    # Configurar reyes negros
    for pos in rey_negro:
        f, c = convertir_posicion(pos)
        TABLERO[f][c] = 'RN'


def mostrar_tablero():
    """Muestra la matriz del tablero de forma legible"""
    print("\nMatriz configurada del tablero:")
    print("     a    b    c    d    e    f    g    h")
    print("  +---------------------------------------+")
    for i, fila in enumerate(TABLERO):
        print(f"{8-i} |", end="")
        for pieza in fila:
            print(f" {pieza} ", end="")
        print(f"| {8-i}")
    print("  +---------------------------------------+")
    print("     a    b    c    d    e    f    g    h")


def cargar_imagen(ruta):
    img = cv2.imread(ruta)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta}")
    return img


def redimensionar_imagen(img):
    """Redimensiona a 360×360 px para que cada casilla recortada quede 100×100."""
    return cv2.resize(img, (720, 720))


def calcular_grilla_completa_9x9(puntos_7x7):
    """
    A partir de 7×7 esquinas internas, construye una grilla 9×9 (intersecciones),
    extrapolando filas y columnas exteriores tal como antes.
    Devuelve array shape=(9,9,2) float32.
    """
    grilla_9x9 = np.zeros((9, 9, 2), dtype=np.float32)
    grilla_9x9[1:8, 1:8] = puntos_7x7

    # Extrapolar fila superior (fila 0)
    for col in range(1, 8):
        vector = puntos_7x7[0, col-1] - puntos_7x7[1, col-1]
        grilla_9x9[0, col] = puntos_7x7[0, col-1] + vector

    # Extrapolar fila inferior (fila 8)
    for col in range(1, 8):
        vector = puntos_7x7[6, col-1] - puntos_7x7[5, col-1]
        grilla_9x9[8, col] = puntos_7x7[6, col-1] + vector

    # Extrapolar columna izquierda (col 0)
    for fila in range(1, 8):
        vector = puntos_7x7[fila-1, 0] - puntos_7x7[fila-1, 1]
        grilla_9x9[fila, 0] = puntos_7x7[fila-1, 0] + vector

    # Extrapolar columna derecha (col 8)
    for fila in range(1, 8):
        vector = puntos_7x7[fila-1, 6] - puntos_7x7[fila-1, 5]
        grilla_9x9[fila, 8] = puntos_7x7[fila-1, 6] + vector

    # Extrapolar las 4 esquinas extremas
    grilla_9x9[0, 0] = grilla_9x9[0, 1] + (grilla_9x9[0, 1] - grilla_9x9[0, 2])
    grilla_9x9[0, 8] = grilla_9x9[0, 7] + (grilla_9x9[0, 7] - grilla_9x9[0, 6])
    grilla_9x9[8, 0] = grilla_9x9[8, 1] + (grilla_9x9[8, 1] - grilla_9x9[8, 2])
    grilla_9x9[8, 8] = grilla_9x9[8, 7] + (grilla_9x9[8, 7] - grilla_9x9[8, 6])

    return grilla_9x9


# probar estoooooooooooooooooooo

ultimos_numeros_usados = {}
def _obtener_siguiente_numero_pieza(pieza_codigo, ruta_carpeta_destino):
    """
    Escanea la carpeta de destino para encontrar el siguiente número disponible
    para una pieza específica, garantizando nombres únicos y persistentes.
    """
    # Si ya calculamos el siguiente número en esta sesión, simplemente lo incrementamos
    if pieza_codigo in ultimos_numeros_usados:
        ultimos_numeros_usados[pieza_codigo] += 1
        return ultimos_numeros_usados[pieza_codigo]

    max_num = 0
    # Patrón para encontrar archivos como "PB_1.jpg", "PN_123.jpg"
    patron = re.compile(rf"^{re.escape(pieza_codigo)}_(\d+)\.jpg$")

    try:
        # Lista los archivos en la carpeta de destino
        for nombre_archivo in os.listdir(ruta_carpeta_destino):
            coincidencia = patron.match(nombre_archivo)
            if coincidencia:
                # Si coincide, extrae el número y lo convierte a entero
                numero_en_archivo = int(coincidencia.group(1))
                if numero_en_archivo > max_num:
                    max_num = numero_en_archivo # Actualiza el número más alto
    except FileNotFoundError:
        # Si la carpeta no existe, max_num se mantiene en 0
        pass

    # Almacena el número más alto encontrado para futuras llamadas en la misma sesión
    ultimos_numeros_usados[pieza_codigo] = max_num + 1
    return ultimos_numeros_usados[pieza_codigo]


def guardar_casilla(imagen, pieza):
    """
    Guarda la imagen en la carpeta correspondiente a la pieza,
    con un nombre único "<pieza>_<numero_siguiente>.jpg".
    """
    if pieza == '--': # Si la pieza es vacía, no guardar
        return

    carpeta_destino_nombre = CARPETAS_PIEZAS.get(pieza, "Vacia")
    ruta_carpeta_destino = RUTA_DATASET / carpeta_destino_nombre
    os.makedirs(ruta_carpeta_destino, exist_ok=True)
    numero = _obtener_siguiente_numero_pieza(pieza, ruta_carpeta_destino)
    nombre_archivo = f"{pieza}_{numero}.jpg"
    ruta_completa_archivo = ruta_carpeta_destino / nombre_archivo

    try:
        # Guarda la imagen usando OpenCV (asumiendo que 'imagen' es un array NumPy de OpenCV)
        cv2.imwrite(str(ruta_completa_archivo), imagen)
        print(f"[GUARDADO] {ruta_completa_archivo}")
    except Exception as e:
        messagebox.showerror("Error al guardar", f"No se pudo guardar la imagen: {e}")


def procesar_tablero(ruta_imagen):
    """
    1) Crea carpetas.
    2) Configura TABLERO desde las listas.
    3) Carga la imagen, la rota 270° (90° CCW), y luego la redimensiona a 360×360.
    4) Detecta esquinas 7×7 del tablero.
    5) Reordena explícitamente esos 7×7 puntos para que queden
       “fila 0=tope” → “fila 6=base”, cada fila de izquierda a derecha.
    6) Extrapola a grilla 9×9 tal como antes.
    7) Recorre 8×8 casillas en orden (a8→…→h8, a7→…→h7, …, a1→h1):
       • recorta,
       • sobrepone texto con “posición algebraica + código de pieza”,
       • muestra cada recorte en pantalla,
       • guarda SOLO si hay pieza (pieza != '--'), numerado indefinidamente.
    """
    crear_estructura_carpetas()
    configurar_tablero_desde_listas()
    mostrar_tablero()

    # ----------------------------
    # BLOQUE 1: cargar y rotar 270°
    # ----------------------------
    img_original = cargar_imagen(ruta_imagen)
    # Rotamos 270° en sentido horario → 90° CCW en OpenCV
    img_rotada_90 = cv2.rotate(img_original, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_rotada_180 = cv2.rotate(img_rotada_90, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # ----------------------------
    # BLOQUE 2: redimensionar
    # ----------------------------
    img = redimensionar_imagen(img_rotada_180)  # ahora 360×360, ya rotada 270°

    # ----------------------------
    # BLOQUE 3: detectar 7×7 esquinas internas
    # ----------------------------
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.equalizeHist(gris)
    patron = (7, 7)
    encontrado, esquinas = cv2.findChessboardCornersSB(gris, patron, cv2.CALIB_CB_EXHAUSTIVE)
    if not encontrado:
        print("Error: No se detectó el tablero de ajedrez")
        return

    # Transformamos a (7,7,2)
    puntos_7x7 = esquinas.reshape(patron[0], patron[1], 2)

    # ─── PASO EXTRA: REORDENAR filas y columnas de puntos_7x7 ───
    # 1) Calcular “media Y” de cada fila
    mean_y_por_fila = [np.mean(puntos_7x7[i, :, 1]) for i in range(7)]
    # 2) Ordenar filas de menor a mayor media Y (arriba→abajo)
    filas_orden = np.argsort(mean_y_por_fila)
    pts_reordenado = puntos_7x7[filas_orden, :, :].copy()
    # 3) En cada fila, ordenar puntos por coordenada X (izquierda→derecha)
    for i in range(7):
        col_idx = np.argsort(pts_reordenado[i, :, 0])
        pts_reordenado[i, :, :] = pts_reordenado[i, col_idx, :]
    puntos_7x7 = pts_reordenado

    # ----------------------------
    # BLOQUE 4: construir grilla 9×9
    # ----------------------------
    grilla_9x9 = calcular_grilla_completa_9x9(puntos_7x7)

    # ----------------------------
    # BLOQUE 5: recortar cada casilla y guardar
    # ----------------------------
    tam_casilla = 100
    img_marcada = img.copy()

    for fila in range(8):
        for col in range(8):
            # (1) Determinar notación algebraica
            letra_col = chr(ord('a') + col)
            num_fila = 8 - fila
            pos_alg = f"{letra_col}{num_fila}"

            # (2) Obtener vértices de la casilla
            pts_origen = np.array([
                grilla_9x9[fila,   col],       # sup. izq.
                grilla_9x9[fila,   col + 1],   # sup. der.
                grilla_9x9[fila + 1, col + 1], # inf. der.
                grilla_9x9[fila + 1, col]      # inf. izq.
            ], dtype="float32")

            # (3) Transformación de perspectiva → recorte
            pts_dest = np.array([
                [0, 0],
                [tam_casilla, 0],
                [tam_casilla, tam_casilla],
                [0, tam_casilla]
            ], dtype="float32")
            M = cv2.getPerspectiveTransform(pts_origen, pts_dest)
            casilla = cv2.warpPerspective(img, M, (tam_casilla, tam_casilla))

            # (4) Obtener código de pieza para esa posición
            pieza = TABLERO[fila][col]

            

            # (6) Mostrar la casilla en pantalla para verificación
            #ventana = f"{pos_alg} ({pieza})"
            #cv2.imshow(ventana, casilla)
            #cv2.waitKey(50)  # 50 ms para avanzar

            # (7) Dibujar contorno coloreado en la imagen marcada
            color = (0, 0, 255) if pieza != '--' else (0, 255, 0)
            pts_int = np.array(pts_origen, dtype=np.int32)
            cv2.polylines(img_marcada, [pts_int], True, color, 2)

            # (8) Guardar solo si hay pieza
            if pieza != '--':
                guardar_casilla(casilla, pieza)

    # Mostrar al final el tablero con todos los bordes coloreados
   # cv2.imshow("Tablero con contornos", img_marcada)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()




#interfaz grafica

# Diccionario de imágenes
imagenes_piezas = {}

# Mapeo de nombres reales a códigos de pieza
nombres_piezas_codigos = {
    "Peón Blanco": "PB", "Peón Negro": "PN",
    "Torre Blanca": "TB", "Torre Negra": "TN",
    "Caballo Blanco": "CB", "Caballo Negro": "CN",
    "Alfil Blanco": "AB", "Alfil Negro": "AN",
    "Dama Blanca": "DB", "Dama Negra": "DN",
    "Rey Blanco": "RB", "Rey Negro": "RN"
}

def cargar_imagenes(size):
    piezas = ['PB', 'PN', 'TB', 'TN', 'CB', 'CN', 'AB', 'AN', 'DB', 'DN', 'RB', 'RN']
    for pieza in piezas:
        ruta = RUTA_BASE_GUI / f"{pieza}.png"
        img = Image.open(ruta).resize((size, size))
        imagenes_piezas[pieza] = ImageTk.PhotoImage(img)

# Ventana de ingreso manual
class VentanaPrincipal(tk.Tk):
    def __init__(self, size=60):
        super().__init__()
        self.title("Toma de Datos del Tablero")
        self.size = size
        self.configure(bg="#f2f2f2")

        cargar_imagenes(self.size)


        # Inicializar la cámara UNA SOLA VEZ
        self.camera = None
        self.camera_available = False
        if cv2:
            self.camera = cv2.VideoCapture(camara) # Intenta abrir la cámara
            if not self.camera.isOpened():
                messagebox.showerror("Error de Cámara", "No se pudo acceder a la cámara. Asegúrese de que no esté en uso y tenga los permisos necesarios.")
                self.camera = None # Asegurarse de que sea None si falla
            else:
                self.camera_available = True
        else:
            messagebox.showerror("Error de Cámara", "La librería 'opencv-python' no está instalada o no se pudo importar.")

        # Configurar protocolo de cierre para liberar la cámara
        self.protocol("WM_DELETE_WINDOW", self.on_closing)


        input_frame = tk.Frame(self, bg="#f2f2f2")
        input_frame.grid(row=0, column=0, sticky="nw", padx=10, pady=10)

        tk.Label(input_frame, text="Tipo de pieza:", bg="#f2f2f2", font=("Inter", 12)).grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.combo_pieza = ttk.Combobox(input_frame, values=list(nombres_piezas_codigos.keys()))
        self.combo_pieza.grid(row=0, column=1, padx=5, pady=5)
        self.combo_pieza.set("Peón Blanco")

        tk.Label(input_frame, text="Posición (ej: c2):", bg="#f2f2f2", font=("Inter", 12)).grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.entry_pos = tk.Entry(input_frame, font=("Inter", 11))
        self.entry_pos.grid(row=1, column=1, padx=5, pady=5)

        tk.Button(input_frame, text="Agregar Pieza", command=self.agregar_pieza,
                  bg="#4CAF50", fg="white", font=("Inter", 11, "bold"), cursor="hand2",
                  relief="raised", bd=3, padx=10, pady=5).grid(row=2, column=0, columnspan=2, pady=15, sticky="ew")

        tk.Button(input_frame, text="Limpiar Tablero", command=self.limpiar_tablero,
                  bg="#FF9800", fg="white", font=("Inter", 11, "bold"), cursor="hand2",
                  relief="raised", bd=3, padx=10, pady=5).grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")
        
        # Nuevo botón "Arreglar al Azar"
        tk.Button(input_frame, text="Arreglar al Azar", command=self.arreglar_al_azar,
                  bg="#9C27B0", fg="white", font=("Inter", 11, "bold"), cursor="hand2",
                  relief="raised", bd=3, padx=10, pady=5).grid(row=4, column=0, columnspan=2, pady=5, sticky="ew")


        self.notation_offset = 30
        self.canvas_width = 8 * self.size + 2 * self.notation_offset
        self.canvas_height = 8 * self.size + 2 * self.notation_offset

        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height,
                                 highlightthickness=2, highlightbackground="#333", bg="#f2f2f2")
        self.canvas.grid(row=0, column=1, rowspan=8, padx=20, pady=10)

        self.dibujar_tablero()

        button_frame = tk.Frame(self, bg="#f2f2f2")
        button_frame.grid(row=8, column=0, columnspan=2, pady=15)

        tk.Button(button_frame, text="Tomar Foto del Tablero", command=self.tomar_foto,
                  bg="#2196F3", fg="white", font=("Inter", 11, "bold"), width=20,
                  relief="raised", bd=3, padx=10, pady=5, cursor="hand2").grid(row=0, column=0, padx=10)

        tk.Button(button_frame, text="Salir", command=self.on_closing,
                  bg="#f44336", fg="white", font=("Inter", 11, "bold"), width=20,
                  relief="raised", bd=3, padx=10, pady=5, cursor="hand2").grid(row=0, column=1, padx=10)

    def dibujar_tablero(self):
        colores = ["#eeeed2", "#769656"]
        self.canvas.delete("all")

        for fila in range(8):
            for col in range(8):
                x1 = col * self.size + self.notation_offset
                y1 = fila * self.size + self.notation_offset
                color = colores[(fila + col) % 2]
                self.canvas.create_rectangle(x1, y1, x1 + self.size, y1 + self.size, fill=color, outline="")

                pieza_codigo = TABLERO[fila][col]
                if pieza_codigo in imagenes_piezas:
                    self.canvas.create_image(x1, y1, image=imagenes_piezas[pieza_codigo], anchor="nw")
                elif pieza_codigo:
                    text_color = "black" if 'B' in pieza_codigo else "white"
                    self.canvas.create_text(x1 + self.size/2, y1 + self.size/2,
                                            text=pieza_codigo, font=("Inter", 16, "bold"), fill=text_color)

        # Nomenclatura columnas (a-h)
        for i in range(8):
            col_char = chr(ord('a') + i)
            self.canvas.create_text(i * self.size + self.size / 2 + self.notation_offset,
                                    self.notation_offset / 2, text=col_char, font=("Inter", 10, "bold"), fill="#333")
            self.canvas.create_text(i * self.size + self.size / 2 + self.notation_offset,
                                    self.canvas_height - self.notation_offset / 2, text=col_char, font=("Inter", 10, "bold"), fill="#333")

        # Nomenclatura filas (1-8)
        for i in range(8):
            row_num = 8 - i
            self.canvas.create_text(self.notation_offset / 2,
                                    i * self.size + self.size / 2 + self.notation_offset, text=str(row_num), font=("Inter", 10, "bold"), fill="#333")
            self.canvas.create_text(self.canvas_width - self.notation_offset / 2,
                                    i * self.size + self.size / 2 + self.notation_offset, text=str(row_num), font=("Inter", 10, "bold"), fill="#333")

    def agregar_pieza(self):
        pieza_nombre = self.combo_pieza.get()
        pos_str = self.entry_pos.get().strip().lower()

        pieza_codigo = nombres_piezas_codigos.get(pieza_nombre)

        if not pieza_codigo:
            messagebox.showerror("Error de Entrada", "Tipo de pieza no válido.")
            return

        if len(pos_str) == 2 and 'a' <= pos_str[0] <= 'h' and '1' <= pos_str[1] <= '8':
            fila, col = convertir_posicion(pos_str)
            TABLERO[fila][col] = pieza_codigo
            self.dibujar_tablero()
            self.entry_pos.delete(0, tk.END)
        else:
            messagebox.showerror("Error de Entrada", "Posición no válida. Usa el formato 'columna+fila' (ej: c2).")

    def limpiar_tablero(self):
        global TABLERO
        TABLERO = [['' for _ in range(8)] for _ in range(8)]
        self.dibujar_tablero()
        

    def arreglar_al_azar(self):
        self.limpiar_tablero() # Limpia el tablero antes de colocar nuevas piezas

        piezas_blancas = [
        #    "RB", "DB", "TB", "TB", "AB", "AB", "CB", "CB",
        #    "PB", "PB"
             "RB", "DB"
        ]
        piezas_negras = [
        #    "RN", "DN", "TN", "TN", "AN", "AN", "CN", "CN",
        #    "PN", "PN"
             "RN", "DN"
        ]

        todas_las_piezas = piezas_blancas + piezas_negras

        # Generar todas las posiciones posibles en el tablero
        posiciones_disponibles = [(f, c) for f in range(8) for c in range(8)]
        random.shuffle(posiciones_disponibles) # Mezclar las posiciones

        # Asignar una posición aleatoria a cada pieza
        for pieza_codigo in todas_las_piezas:
            if posiciones_disponibles:
                fila, col = posiciones_disponibles.pop(0) # Tomar una posición aleatoria
                TABLERO[fila][col] = pieza_codigo
            else:
                messagebox.showwarning("Advertencia", "No hay suficientes casillas para todas las piezas.")
                break
        
        self.dibujar_tablero()
        


    def tomar_foto(self):


        filename = f"tablero_ajedrez.png"

        if not self.camera_available or self.camera is None:
            messagebox.showerror("Error de Cámara", "La cámara no está disponible o no se pudo inicializar.")
            return

        ret, frame = self.camera.read() # Solo leer un frame de la cámara ya abierta

        if ret:
            try:
                # Convertir de BGR (formato de OpenCV) a RGB (formato de PIL)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                img.save(filename)
                messagebox.showinfo("Foto Guardada", f"La foto de la cámara se ha guardado como:\n{filename}")
                ruta_tablero = RUTA_BASE / "tablero_ajedrez.png"
                procesar_tablero(ruta_tablero)

            except Exception as e:
                messagebox.showerror("Error al tomar foto de cámara", f"Ocurrió un error al guardar la foto: {e}")
       

    def on_closing(self):
        """
        Libera los recursos de la cámara antes de cerrar la ventana.
        """
        if self.camera:
            self.camera.release()
        self.destroy()


# Ejecutar
if __name__ == "__main__":
    app = VentanaPrincipal()
    app.mainloop()


#if __name__ == "__main__":
#    ruta_tablero = (
#        r"C:\\Users\\juanu\\Downloads\\UR5\\test.jpg"
#    )
#    try:
#        procesar_tablero(ruta_tablero)
#    except Exception as e:
#        print("Error:", e)
#        traceback.print_exc()
#        cv2.destroyAllWindows()