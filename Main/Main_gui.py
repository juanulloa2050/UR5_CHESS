import Move_engine as move_engine
import CNN as vision_engine
import Chess_engine as chess_engine

from tkinter import messagebox, ttk
import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path
import cv2
import chess
import time

# === VARIABLES GLOBALES ===
color_seleccionado = None
dificultad_seleccionada = None
Historial_jugadas = []  # Lista para almacenar el historial de jugadas
camara = 0  # Cambiar a 1 si se quiere usar la cámara

FILES = "abcdefgh"

        # Definir el estado inicial del tablero si no existe
board_matrix = [
        ["r", "n", "b", "q", "k", "b", "n", "r"],
        ["p", "p", "p", "p", "p", "p", "p", "p"],
        ["empty"] * 8,
        ["empty"] * 8,
        ["empty"] * 8,
        ["empty"] * 8,
        ["P", "P", "P", "P", "P", "P", "P", "P"],
        ["R", "N", "B", "Q", "K", "B", "N", "R"]
    ]



BASE_DIR = Path(__file__).resolve().parent.parent

# === LÓGICA PREVIA AL INICIO DE LA GUI ===
def inicializar_app(app):
    print(" Cargando configuraciones previas antes de mostrar la app")
    #move_engine.iniciar_robot()  # Inicia el robot UR5 y la gripper Robotiq

    try:
        app.camera = cv2.VideoCapture(camara)
        if not app.camera.isOpened():
            app.camera.release()
            app.camera = None
            app.camera_available = False
            messagebox.showerror("Error de Cámara", "No se pudo acceder a la cámara. Asegúrese de que no esté en uso y tenga los permisos necesarios.")
        else:
            app.camera_available = True
            print(" Cámara inicializada correctamente")
    except Exception as e:
        app.camera = None
        app.camera_available = False
        messagebox.showerror("Error de Cámara", f"No se pudo iniciar la cámara: {e}")
    
    



class ConfiguracionInicial(tk.Frame):
    def __init__(self, master, iniciar_callback):
        super().__init__(master)
        self.master = master
        self.iniciar_callback = iniciar_callback
        self.configure(bg="#f5f5f5")

        estilo = ttk.Style()
        estilo.theme_use("default")
        estilo.configure("TButton",
                         font=("Helvetica", 11, "bold"),
                         background="#27ae60",
                         foreground="white",
                         padding=6)
        estilo.map("TButton", background=[("active", "#219150")])
        estilo.configure("TMenubutton", font=("Helvetica", 10), background="white", foreground="#333")

        tk.Label(self, text="UR CHESS", font=("Helvetica", 28, "bold"), fg="#2c3e50", bg="#f5f5f5").pack(pady=(20, 10))

        contenedor = tk.Frame(self, bg="white", bd=2, relief="ridge")
        contenedor.pack(pady=10)

        tk.Label(contenedor, text="Configuración de Partida", font=("Helvetica", 16, "bold"),
                 bg="white", fg="#222").pack(pady=(15, 10))

        self.color_var = tk.StringVar(value="Blanco")
        frame_color = tk.LabelFrame(contenedor, text="Color del Jugador", font=("Helvetica", 11, "bold"),
                                    bg="white", fg="#444", bd=1, padx=10, pady=10)
        frame_color.pack(pady=10, padx=20, fill="x")

        estilos_rb = {
            "font": ("Helvetica", 10),
            "indicatoron": 0,
            "width": 10,
            "bd": 0,
            "relief": "flat",
            "bg": "#e0e0e0",
            "activebackground": "#d0d0d0",
            "cursor": "hand2",
            "selectcolor": "#a0a0a0"
        }

        for color in ["Blanco", "Negro"]:
            rb = tk.Radiobutton(frame_color, text=color, variable=self.color_var, value=color, **estilos_rb)
            rb.pack(side="left", padx=10, pady=5)

        self.dificultad_var = tk.StringVar(value="Media")
        frame_dificultad = tk.Frame(contenedor, bg="white")
        tk.Label(frame_dificultad, text="Dificultad:", font=("Helvetica", 11), bg="white").pack(side="left", padx=5)

        dificultad_menu = ttk.OptionMenu(frame_dificultad, self.dificultad_var, self.dificultad_var.get(),
                                         "Muy Fácil", "Fácil", "Media", "Difícil", "Muy Difícil")
        dificultad_menu.pack(side="left", padx=5)

        self.tooltip = tk.Label(contenedor, text="", font=("Helvetica", 9, "italic"), bg="white", fg="#666")
        self.tooltip.pack()
        dificultad_menu.bind("<Enter>", lambda e: self.tooltip.config(text="Selecciona el nivel de dificultad del oponente"))
        dificultad_menu.bind("<Leave>", lambda e: self.tooltip.config(text=""))

        frame_dificultad.pack(pady=15)

        boton = ttk.Button(contenedor, text="🚀 Iniciar Partida", command=self.iniciar_partida, style="TButton")
        boton.pack(pady=20)

        ruta_gui = BASE_DIR / "Imagenes_GUI"

        try:
            self.logo_izq = ImageTk.PhotoImage(Image.open(ruta_gui / "logo1.jpg").resize((100, 100)))
            self.logo_der = ImageTk.PhotoImage(Image.open(ruta_gui / "Unisabana.png").resize((100, 100)))
            tk.Label(self, image=self.logo_izq, bg="#f5f5f5").place(relx=0.0, rely=1.0, anchor="sw", x=10, y=-10)
            tk.Label(self, image=self.logo_der, bg="#f5f5f5").place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)
        except Exception as e:
            print("No se pudieron cargar los logos:", e)

    def iniciar_partida(self):
        global color_seleccionado, dificultad_seleccionada
        color_seleccionado = self.color_var.get()
        dificultad_seleccionada = self.dificultad_var.get()
        self.iniciar_callback(color_seleccionado, dificultad_seleccionada)


class TableroAjedrez(tk.Frame):
    def __init__(self, master, salir_callback):
        super().__init__(master)
        self.master = master
        self.salir_callback = salir_callback
        self.estado_actual = "esperando"
        self.maquina_activa = False
        self.configure(bg="white")

        tk.Label(self, text="UR CHESS", font=("Helvetica", 24, "bold"), fg="#2c3e50", bg="white").pack(pady=(10, 5))

        main_frame = tk.Frame(self, bg="white")
        main_frame.pack(pady=5)

        tablero_frame = tk.Frame(main_frame, bg="#ffffff", bd=2, relief="solid")
        tablero_frame.pack(side="left", padx=(20, 10))

        self.canvas = tk.Canvas(tablero_frame, width=600, height=620, bg="#ffffff", highlightthickness=0)
        self.canvas.pack()
        self.dibujar_tablero()

        panel_derecho = tk.Frame(main_frame, bg="white")
        panel_derecho.pack(side="left", padx=(10, 20), fill="y")

        self.lista_jugadas = tk.Text(panel_derecho, width=25, height=20, state='disabled',
                                     bg="#333", fg="white", font=("Courier", 10))
        self.lista_jugadas.pack(pady=(0, 10))

        # botón: Finalizar jugada
        tk.Button(panel_derecho, text="✔ Finalizar Jugada", bg="#4CAF50", fg="white",
                  font=("Helvetica", 12), command=self.finalizar_jugada, cursor="hand2").pack(pady=(5, 2), fill="x")

        # Botón: Pedir Pista
        tk.Button(panel_derecho, text="💡 Pedir Pista", bg="#2196F3", fg="white",
                  font=("Helvetica", 12), command=self.pedir_pista, cursor="hand2").pack(pady=2, fill="x")

        # Botón: Rendirse
        tk.Button(panel_derecho, text="🏳 Rendirse", bg="#f44336", fg="white",
                  font=("Helvetica", 12), command=self.salir_callback, cursor="hand2").pack(pady=5, fill="x")


        #ruta_gui = Path("./Imagenes_GUI")
        ruta_gui = BASE_DIR / "Imagenes_GUI"
        try:
            logos_frame = tk.Frame(panel_derecho, bg="white")
            self.logo1 = ImageTk.PhotoImage(Image.open(ruta_gui / "logo1.jpg").resize((80, 80)))
            self.logo2 = ImageTk.PhotoImage(Image.open(ruta_gui / "Unisabana.png").resize((80, 80)))
            tk.Label(logos_frame, image=self.logo1, bg="white").pack(side="left", padx=5)
            tk.Label(logos_frame, image=self.logo2, bg="white").pack(side="left", padx=5)
            logos_frame.pack(side="bottom", pady=(30, 10))
        except Exception as e:
            print("No se pudieron cargar los logos:", e)

        self.master.geometry("900x660")
        self.master.update_idletasks()
        ancho = 900
        alto = 660
        x = (self.master.winfo_screenwidth() // 2) - (ancho // 2)
        y = (self.master.winfo_screenheight() // 2) - (alto // 2)
        self.master.geometry(f"{ancho}x{alto}+{x}+{y}")

    def iniciar_maquina_estados(self):
        if not self.maquina_activa:
            self.maquina_activa = True
            self.dibujar_tablero(board_matrix)
            self.tomar_foto()
            self.chess_engine = chess_engine.ChessEngine(player_color="black", player_elo=1400, stockfish_elo=2200)
            self.vision_engine = vision_engine.VisionEngine()

            self.actualizar_estado()

    def actualizar_estado(self):
        if self.estado_actual == "jugando":
            move_engine.take_picture()  # Tomar foto del tablero actual
            self.tomar_foto()

            move_engine.go_home()  # Mover el robot a la posición inicial
            #move_engine.picture_to_home()
            fen, board_matrix, imagen_procesada = self.vision_engine.procesar_imagen(mostrar=False)
            print (board_matrix)

            self.dibujar_tablero(board_matrix)
            self.update_idletasks()
            time.sleep(0.5)  # Esperar a que se dibuje el tablero

            best_move, is_capture, is_castling, game_status = self.chess_engine.get_best_move(fen)
            print(self.es_captura(board_matrix, best_move))
            is_capture = self.es_captura(board_matrix, best_move)
            board_matrix = self.aplicar_jugada(board_matrix, best_move)
            self.dibujar_tablero(board_matrix)

            
            if best_move:
                print("Mejor jugada:", best_move)
                print("Captura:", is_capture, "| Enroque:", is_castling)

            casillainicial = chess.square_name(best_move.from_square)
            casillafinal = chess.square_name(best_move.to_square)
            
            
            if is_capture == True:
                print("¡Es captura!")
                move_engine.take(casillafinal) 
                print("¡Captura realizada!")

            move_engine.move_pieza(casillainicial, casillafinal)
            move_engine.go_home()

            self.estado_actual = "esperando"
            pass
        elif self.estado_actual == "esperando":

            pass

        if self.maquina_activa:
            self.after(100, self.actualizar_estado)

    def dibujar_tablero(self, board_matrix=None, color_seleccionado=None):
        """
        Redibuja el tablero en self.canvas.

        Si no se proporcionan board_matrix o color_seleccionado,
        se usan los atributos self.board_matrix y self.color_seleccionado.
        """

        # ──────── 0. Entrada de datos ────────
        if board_matrix is None:
            board_matrix = getattr(self, "board_matrix",
                                   [["empty"]*8 for _ in range(8)])
        if color_seleccionado is None:
            color_seleccionado = getattr(self, "color_seleccionado", "Blanco")

        # ──────── 1. Constantes ──────────────
        CASILLA  = 60
        OFFSET   = 60
        COLORES  = ("#f0d9b5", "#b58863")
        UNICODE_PIECES = {
            "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
            "K": "♚", "Q": "♛","R": "♜", "B": "♝", "N": "♞", "P": "♟",
    "       empty": "·"
        }
        invertido = (color_seleccionado == "Negro")

        # ──────── 2. Limpiar y dibujar casillas ────────
        self.canvas.delete("all")
        for fila in range(8):
            for col in range(8):
                xb = col if not invertido else 7 - col
                yb = fila if not invertido else 7 - fila
                x1 = xb*CASILLA + OFFSET
                y1 = yb*CASILLA + OFFSET
                color = COLORES[(fila + col) % 2]
                self.canvas.create_rectangle(x1, y1, x1 + CASILLA,
                                             y1 + CASILLA,
                                             fill=color, outline="")

        # ──────── 3. Coordenadas ─────────────
        for i in range(8):
            letra = chr(ord('a') + (i if not invertido else 7 - i))
            x = i*CASILLA + OFFSET + CASILLA//2
            self.canvas.create_text(x, OFFSET//2, text=letra,
                                    font=("Arial", 12, "bold"))
            self.canvas.create_text(x, OFFSET + 8*CASILLA + OFFSET//4,
                                    text=letra, font=("Arial", 12, "bold"))

            num = str(8 - i) if not invertido else str(i + 1)
            y = i*CASILLA + OFFSET + CASILLA//2
            self.canvas.create_text(OFFSET//2, y, text=num,
                                    font=("Arial", 12, "bold"))
            self.canvas.create_text(OFFSET + 8*CASILLA + OFFSET//4, y,
                                    text=num, font=("Arial", 12, "bold"))

        # ──────── 4. Piezas ─────────────
        fuente_piezas = ("Arial", int(CASILLA * 0.75))
        for fila in range(8):
            for col in range(8):
                pieza = board_matrix[fila][col]
                if pieza == "empty":
                    continue
                xb = col if not invertido else 7 - col
                yb = fila if not invertido else 7 - fila
                cx = xb*CASILLA + OFFSET + CASILLA//2
                cy = yb*CASILLA + OFFSET + CASILLA//2
                color_text = "white" if pieza.isupper() else "black"
                self.canvas.create_text(cx, cy,
                                        text=UNICODE_PIECES[pieza],
                                        font=fuente_piezas,
                                        fill=color_text,
                                        tags="piezas")

        # (Opcional) refresco inmediato si llamas desde un hilo externo
        # self.canvas.update_idletasks()




    def pedir_pista(self):
        messagebox.showinfo("Pista", "Aquí iría la sugerencia de jugada.")
    
    def finalizar_jugada(self):
        #messagebox.showinfo("Turno Finalizado", "Has finalizado tu turno.")
        self.estado_actual = "jugando"

    def tomar_foto(self):
        if self.master.camera_available:
            # Espera a que la cámara se estabilice (opcional)
            cv2.waitKey(500)

            ret, frame = self.master.camera.read()
            if ret:
                nombre = f"UR5jugada_actual.jpg"
                base_dir = Path(__file__).resolve().parent          # carpeta donde vive tu .py
                foto_path = base_dir / "jugada_actual.png" # usa .png como en el resto del proyecto

                cv2.imwrite(str(foto_path), frame)
                #messagebox.showinfo("Foto tomada", f"Se guardó como {nombre}")
            else:
                messagebox.showerror("Error", "No se pudo leer imagen de la cámara.")
        else:
            messagebox.showerror("Cámara", "Cámara no disponible.")

    def sq_to_rc(square: str):
        """
        Convierte 'a1'→(7,0), 'h8'→(0,7) para nuestra matriz:
        fila 0 = 8ª fila, fila 7 = 1ª fila.
        """
        file = FILES.index(square[0])
        rank = 8 - int(square[1])
        return rank, file


    @staticmethod
    def aplicar_jugada(board, move):
        """
        Acepta:
            • move como str  -> "c2c3", "e7e8q", "e1g1" ...
            • move como chess.Move (python-chess)
        Modifica board in-place y lo devuelve.
        """
        # ── 1) Convierte a cadena UCI si es necesario ──────────────────
        if isinstance(move, chess.Move):
            move = move.uci()          # p.ej. chess.Move.from_uci("c2c3") → "c2c3"

        #move = move.strip().lower()

        # ── 2) Parseo origen, destino, promoción ───────────────────────
        from_sq = move[:2]
        to_sq   = move[2:4]
        promo   = move[4:]                 # '' si no hay promo

        fr, fc = TableroAjedrez.sq_to_rc(from_sq)
        tr, tc = TableroAjedrez.sq_to_rc(to_sq)

        pieza = board[fr][fc]
        board[fr][fc] = "empty"

        # ── 3) En-passant ──────────────────────────────────────────────
        if pieza.lower() == "p" and fc != tc and board[tr][tc] == "empty":
            paso_r = tr + (1 if pieza.isupper() else -1)
            board[paso_r][tc] = "empty"

        # ── 4) Promoción ───────────────────────────────────────────────
        if promo:
            pieza = promo if pieza.islower() else promo.upper()

        board[tr][tc] = pieza

        # ── 5) Enroques ────────────────────────────────────────────────
        if pieza.lower() == "k" and abs(fc - tc) == 2:
            rook_from_c, rook_to_c = (7, 5) if tc == 6 else (0, 3)
            rook_r = fr
            board[rook_r][rook_to_c] = board[rook_r][rook_from_c]
            board[rook_r][rook_from_c] = "empty"

        return board


    @staticmethod
    def es_captura(board, move):
        """
        board: lista 8×8 con piezas ("P","r", "empty", etc.)
        move: chess.Move o UCI string, p.ej. "e2e4", "d7d8q"
        Devuelve True si en la casilla destino había una pieza, 
        False en caso contrario.
        """
        # 1) Normaliza a UCI string
        if isinstance(move, chess.Move):
            move = move.uci()
        move = move.strip().lower()

        # 2) Extrae destino
        to_sq = move[2:4]
        tr, tc = TableroAjedrez.sq_to_rc(to_sq)

        # 3) Captura normal si destino ≠ "empty"
        return board[tr][tc] != "empty"





class AplicacionAjedrez(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("♟ Sistema de Ajedrez")
        self.geometry("500x300")
        self.resizable(False, False)
        self.config(bg="white")
        self.frame_actual = None

        #camara
        self.camera = None
        self.camera_available = False

        inicializar_app(self)  # 🧠 Se ejecuta antes de mostrar GUI

        self.mostrar_frame_configuracion()

        # Manejo de cierre
        self.protocol("WM_DELETE_WINDOW", self.on_closing)


    def mostrar_frame_configuracion(self):
        if self.frame_actual:
            self.frame_actual.destroy()
        self.geometry("500x550")
        self.frame_actual = ConfiguracionInicial(self, self.mostrar_frame_tablero)
        self.frame_actual.pack(expand=True, fill="both")


    def mostrar_frame_tablero(self, color, dificultad):
        if self.frame_actual:
            self.frame_actual.destroy()
        self.geometry("960x760")
        self.frame_actual = TableroAjedrez(self, self.mostrar_frame_configuracion)
        self.frame_actual.pack(expand=True, fill="both")

        self.frame_actual.estado_actual = "esperando"
        self.frame_actual.iniciar_maquina_estados()  # Inicia la máquina de estados
    
    def on_closing(self):
        if self.camera:
            self.camera.release()
            print("📷 Cámara liberada correctamente")
        self.destroy()



if __name__ == "__main__":
    app = AplicacionAjedrez()
    app.mainloop()
