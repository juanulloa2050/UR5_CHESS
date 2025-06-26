import urx
import numpy as np
import math3d as m3d
import time
import sys
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper  # Importar controlador oficial

# -------------------------
# CONFIGURACIÓN DEL ROBOT
# -------------------------
ROBOT_IP = "192.168.0.100"  # ¡VERIFICAR IP REAL!
robot = None

try:
    print(f"Conectando al robot en {ROBOT_IP}...")
    # Usar RTDE y timeout extendido
    robot = urx.Robot(ROBOT_IP, use_rt=True, timeout=5.0)
    print("¡Conexión exitosa con RTDE!")
    
except Exception as e:
    print(f"\nERROR: {e}")
    print("Soluciones posibles:")
    print("1. Actualiza la biblioteca: pip install --upgrade urx")
    print("2. Verifica el firmware del robot (debe ser compatible con RTDE)")
    print("3. Usa IP correcta y modo 'Control Remoto' activado")
    sys.exit(1)

# -------------------------
# CONFIGURACIÓN GRIPPER (ESTILO OFICIAL)
# -------------------------
gripper = Robotiq_Two_Finger_Gripper(robot)

def gripper_open():
    gripper.open_gripper()
    time.sleep(0.5)

def gripper_close():
    gripper.close_gripper()
    time.sleep(0.5)

# Coordenadas físicas de la esquina inferior izquierda (h8) y superior derecha (a1)
p1 = np.array([0.300, 0.200, 0.100])  # h8 (origen físico del tablero)
p2 = np.array([0.500, 0.400, 0.100])  # a1 (esquina opuesta)

altura_z_sobre_casilla = 0.100  # Altura segura sobre el tablero
altura_z_agarre = 0.030         # Altura para agarrar piezas
altura_home = m3d.Vector([0.400, 0.000, 0.400])  # Posición de home segura

# Cálculo de desplazamientos por casilla
dx = (p2[0] - p1[0]) / 7.0
dy = (p2[1] - p1[1]) / 7.0

# -------------------------
# GENERACIÓN DE CASILLAS
# -------------------------

# Orden invertido: h8 como origen
columnas = "hgfedcba"
filas = "87654321"

casillas = {}
for i in range(8):  # columnas h–a
    for j in range(8):  # filas 8–1
        nombre = f"{columnas[i]}{filas[j]}"
        x = p1[0] + i * dx
        y = p1[1] + j * dy
        casillas[nombre] = [x, y]

# -------------------------
# CONTROL DE LA PINZA ROBOTIQ
# -------------------------

def gripper_open():
    robot.send_program("rq_open()")
    time.sleep(1)

def gripper_close():
    robot.send_program("rq_close()")
    time.sleep(1)

def gripper_set(pos=255, speed=150, force=100):
    # Posición: 0 (abierta) – 255 (cerrada)
    # Velocidad y fuerza: 0 – 255
    script = f"rq_move({pos}, {speed}, {force})"
    robot.send_program(script)
    time.sleep(1)

# -------------------------
# FUNCIONES DE MOVIMIENTO
# -------------------------

def move_to(x, y, z, wait=True):
    pose = m3d.Transform()
    pose.pos = m3d.Vector([x, y, z])
    robot.movel(pose, acc=0.2, vel=0.2, wait=wait)

def go_home():
    pose = [-0.444212,0.30077, 0.58476, 0.156, 0.361, 5.933]
    robot.movel(pose, acc=0.2, vel=0.2, wait=True)

def pick(casilla):
    x, y = casillas[casilla]
    move_to(x, y, altura_z_sobre_casilla)
    move_to(x, y, altura_z_agarre)
    gripper_close()
    move_to(x, y, altura_z_sobre_casilla)

def place(casilla):
    x, y = casillas[casilla]
    move_to(x, y, altura_z_sobre_casilla)
    move_to(x, y, altura_z_agarre)
    gripper_open()
    move_to(x, y, altura_z_sobre_casilla)


# -------------------------
# EJECUCIÓN PRINCIPAL
# -------------------------
if __name__ == "__main__":
    try:
        if robot:
            print("Probando gripper...")
            gripper_open()
            gripper_close()
            
            # Mover a home si está definido
            # go_home()
    finally:
        if robot:
            robot.close()