# Cambiado para crear dataset con anotaciones Pascal VOC para Faster R-CNN
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
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Configuración global de rutas
RUTA_BASE = Path("../Data_set")
RUTA_BASE_GUI = Path("../Imagenes_GUI")
RUTA_DATASET = RUTA_BASE / "data_set_faster_r_cnn"  

# Subcarpetas para el nuevo dataset
RUTA_IMAGENES = RUTA_DATASET / "images"
RUTA_ANOTACIONES = RUTA_DATASET / "annotations"

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
reina_blanca   = ['b2']
reina_negra    = ["a1"]
rey_blanco     = ["g3"]
rey_negro      = ["g4"]

# Mapeo de piezas a nombres de clase
CLASES_PIEZAS = {
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

# Contador global de imágenes para nombres únicos
contador_imagenes = 0


def crear_estructura_carpetas():
    """Crea la estructura de carpetas para el dataset Pascal VOC"""
    RUTA_IMAGENES.mkdir(parents=True, exist_ok=True)
    RUTA_ANOTACIONES.mkdir(parents=True, exist_ok=True)


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
    """Redimensiona a 720×720 px para mantener alta resolución"""
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


def crear_anotacion_voc(imagen, objetos, ruta_imagen, ruta_xml):
    """
    Crea un archivo de anotación en formato Pascal VOC
    
    Args:
        imagen: Imagen OpenCV (para obtener dimensiones)
        objetos: Lista de diccionarios con:
            {'clase': str, 'bbox': (xmin, ymin, xmax, ymax)}
        ruta_imagen: Ruta relativa de la imagen
        ruta_xml: Ruta completa para guardar el XML
    """
    altura, ancho, canales = imagen.shape
    
    # Crear estructura XML
    root = ET.Element("annotation")
    
    # Información de la carpeta
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = os.path.basename(ruta_imagen)
    ET.SubElement(root, "path").text = str(ruta_imagen)
    
    # Información de la fuente
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Chess Dataset"
    
    # Información de tamaño
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(ancho)
    ET.SubElement(size, "height").text = str(altura)
    ET.SubElement(size, "depth").text = str(canales)
    
    ET.SubElement(root, "segmented").text = "0"
    
    # Añadir objetos
    for obj in objetos:
        object_elem = ET.SubElement(root, "object")
        ET.SubElement(object_elem, "name").text = obj['clase']
        ET.SubElement(object_elem, "pose").text = "Unspecified"
        ET.SubElement(object_elem, "truncated").text = "0"
        ET.SubElement(object_elem, "difficult").text = "0"
        
        bbox = ET.SubElement(object_elem, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(int(obj['bbox'][0]))
        ET.SubElement(bbox, "ymin").text = str(int(obj['bbox'][1]))
        ET.SubElement(bbox, "xmax").text = str(int(obj['bbox'][2]))
        ET.SubElement(bbox, "ymax").text = str(int(obj['bbox'][3]))
    
    # Formatear y guardar XML
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open(ruta_xml, "w") as f:
        f.write(xml_str)


def procesar_tablero(ruta_imagen):
    """
    Procesa el tablero para generar anotaciones Pascal VOC:
    1. Crea carpetas para el dataset
    2. Configura TABLERO desde las listas
    3. Carga y procesa la imagen
    4. Detecta esquinas del tablero
    5. Construye grilla 9x9
    6. Genera bounding boxes para cada pieza
    7. Guarda imagen completa y archivo de anotaciones
    """
    global contador_imagenes
    
    crear_estructura_carpetas()
    configurar_tablero_desde_listas()
    mostrar_tablero()

    # Cargar y procesar imagen
    img_original = cargar_imagen(ruta_imagen)
    img_rotada_90 = cv2.rotate(img_original, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_rotada_180 = cv2.rotate(img_rotada_90, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = redimensionar_imagen(img_rotada_180)

    # Detectar esquinas del tablero
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.equalizeHist(gris)
    patron = (7, 7)
    encontrado, esquinas = cv2.findChessboardCornersSB(gris, patron, cv2.CALIB_CB_EXHAUSTIVE)
    if not encontrado:
        print("Error: No se detectó el tablero de ajedrez")
        return

    puntos_7x7 = esquinas.reshape(patron[0], patron[1], 2)

    # Reordenar puntos
    mean_y_por_fila = [np.mean(puntos_7x7[i, :, 1]) for i in range(7)]
    filas_orden = np.argsort(mean_y_por_fila)
    pts_reordenado = puntos_7x7[filas_orden, :, :].copy()
    for i in range(7):
        col_idx = np.argsort(pts_reordenado[i, :, 0])
        pts_reordenado[i, :, :] = pts_reordenado[i, col_idx, :]
    puntos_7x7 = pts_reordenado

    # Construir grilla 9×9
    grilla_9x9 = calcular_grilla_completa_9x9(puntos_7x7)

    # Generar bounding boxes y clases para cada pieza
    objetos = []
    for fila in range(8):
        for col in range(8):
            pieza = TABLERO[fila][col]
            if pieza == '--':
                continue

            # Obtener coordenadas de la casilla
            pts_origen = np.array([
                grilla_9x9[fila,   col],       # sup. izq.
                grilla_9x9[fila,   col + 1],   # sup. der.
                grilla_9x9[fila + 1, col + 1], # inf. der.
                grilla_9x9[fila + 1, col]      # inf. izq.
            ], dtype="float32")

            # Calcular bounding box (axis-aligned)
            xmin = min(pts_origen[:, 0])
            ymin = min(pts_origen[:, 1])
            xmax = max(pts_origen[:, 0])
            ymax = max(pts_origen[:, 1])

            # Añadir objeto a la lista
            objetos.append({
                'clase': CLASES_PIEZAS[pieza],
                'bbox': (xmin, ymin, xmax, ymax)
            })

    max_num = 0
    patron = re.compile(r"tablero_(\d{4})\.jpg$")
    for archivo in os.listdir(RUTA_IMAGENES):
        coincidencia = patron.match(archivo)
        if coincidencia:
            num = int(coincidencia.group(1))
            if num > max_num:
                max_num = num
    
    nombre_archivo = f"tablero_{max_num+1:04d}"
    ruta_imagen_completa = RUTA_IMAGENES / f"{nombre_archivo}.jpg"
    ruta_xml = RUTA_ANOTACIONES / f"{nombre_archivo}.xml"
    
    # Guardar imagen procesada
    cv2.imwrite(str(ruta_imagen_completa), img)
    
    # Crear y guardar anotación VOC
    crear_anotacion_voc(img, objetos, ruta_imagen_completa, ruta_xml)
    
    print(f"[DATASET] Imagen y anotación guardadas: {nombre_archivo}")


""""
    # ------ INICIO CÓDIGO DE VISUALIZACIÓN ------
    # Generar bounding boxes y clases para cada pieza
    objetos = []
    img_mostrar = img.copy()  # Copia para visualización
    
    for fila in range(8):
        for col in range(8):
            pieza = TABLERO[fila][col]
            if pieza == '--':
                continue

            # Obtener coordenadas de la casilla
            pts_origen = np.array([
                grilla_9x9[fila,   col],       # sup. izq.
                grilla_9x9[fila,   col + 1],   # sup. der.
                grilla_9x9[fila + 1, col + 1], # inf. der.
                grilla_9x9[fila + 1, col]      # inf. izq.
            ], dtype="float32")

            # Calcular bounding box (axis-aligned)
            xmin = min(pts_origen[:, 0])
            ymin = min(pts_origen[:, 1])
            xmax = max(pts_origen[:, 0])
            ymax = max(pts_origen[:, 1])

            # Añadir objeto a la lista
            objetos.append({
                'clase': CLASES_PIEZAS[pieza],
                'bbox': (xmin, ymin, xmax, ymax)
            })
            
            # ------ AQUÍ EMPIEZA EL NUEVO CÓDIGO PARA VISUALIZACIÓN ------
            # Dibujar recuadro en la imagen de visualización
            color = (0, 255, 0)  # Verde para las piezas
            grosor = 2
            cv2.rectangle(img_mostrar, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, grosor)
            
            # Añadir etiqueta de pieza
            texto = f"{CLASES_PIEZAS[pieza]}"
            pos_texto = (int(xmin) + 5, int(ymin) + 20)
            cv2.putText(img_mostrar, texto, pos_texto, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, grosor)
            
            # Dibujar puntos de las esquinas
            for punto in pts_origen:
                cv2.circle(img_mostrar, (int(punto[0]), int(punto[1])), 4, (0, 0, 255), -1)

    # Mostrar imagen con recuadros
    ventana = "Verificación de bounding boxes (Presiona cualquier tecla para continuar)"
    cv2.imshow(ventana, img_mostrar)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # ------ FIN DEL CÓDIGO DE VISUALIZACIÓN ------
"""
 


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
        TABLERO = [['--' for _ in range(8)] for _ in range(8)]
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

            except Exception as e:
                messagebox.showerror("Error al tomar foto de cámara", f"Ocurrió un error al guardar la foto: {e}")

        procesar_tablero(ruta_tablero)

    def on_closing(self):
        """
        Libera los recursos de la cámara antes de cerrar la ventana.
        """
        if self.camera:
            self.camera.release()
        self.destroy()


#"""
# Ejecutar
if __name__ == "__main__":
    app = VentanaPrincipal()
    app.mainloop()
#"""
"""
if __name__ == "__main__":
    ruta_tablero = (
        r"C:\\Users\\juanu\\Downloads\\UR5\\tablero_ajedrez.png"  # Ruta de la imagen del tablero
    )
    try:
        procesar_tablero(ruta_tablero)
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        cv2.destroyAllWindows()
"""