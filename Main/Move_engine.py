# Engine de movimiento para el robot UR5 con gripper Robotiq
import numpy as np
import matplotlib.pyplot as plt
import time
import socket
import struct
import math

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

def get_actual_tcp_pose():
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
        current = get_actual_tcp_pose()
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
    move_to(pose_mm_rad=[-484, 288, 527, 3.071, -0.901, -0.07], a=1.2, v=0.5)

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

def iniciar_robot():
    """
    Inicializa el robot UR5 y la gripper Robotiq.
    """
    print("Iniciando robot...")
    gripper_activate()
    gripper_open()
    picture_to_home()
    print("Robot listo para operar.")

def take(casilla):
    """
    Toma una pieza de la casilla especificada.
    """
    pick(casilla)
    X, Y = transform_point(-100, 200)
    move_to(pose_mm_rad=[X, Y , z_sobre, rx, ry, rz], a=1.2, v=0.5) 
    move_to(pose_mm_rad=[X, Y, z_agarre, rx, ry, rz], a=1.2, v=0.5)
    gripper_open()
    move_to(pose_mm_rad=[X, Y, z_sobre, rx, ry, rz], a=1.2, v=0.5)
    gripper_open()

def read_digital_inputs(timeout=5, max_attempts=5):
    """
    Lee las entradas digitales del robot
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect((ip, 30002))
        buffer = b''

        for _ in range(max_attempts):
            try:
                data = sock.recv(4096)
                if not data:
                    time.sleep(0.1)
                    continue

                buffer += data
                offset = 0
                while offset < len(buffer):
                    if len(buffer) - offset < 5:
                        break
                    
                    pack_length = struct.unpack_from(">i", buffer, offset)[0]
                    message_type = buffer[offset + 4]

                    if len(buffer) - offset < pack_length:
                        break

                    if message_type == 16:
                        sub_offset = offset + 5
                        while sub_offset < offset + pack_length:
                            if len(buffer) - sub_offset < 5:
                                break

                            sub_length, sub_type = struct.unpack_from(">IB", buffer, sub_offset)
                            if len(buffer) - sub_offset < sub_length:
                                break

                            if sub_type == 3:
                                digital_input_bits, _ = struct.unpack_from(">II", buffer, sub_offset + 5)
                                return [(digital_input_bits >> bit) & 1 for bit in range(8)]

                            sub_offset += sub_length

                    offset += pack_length
                buffer = buffer[offset:]

            except (socket.timeout, struct.error, IndexError):
                time.sleep(0.1)
                continue

    except socket.timeout:
        print(f"[ERROR] Connection timed out after {timeout} seconds.")
    except ConnectionRefusedError:
        print(f"[ERROR] Connection refused. Is the robot reachable at {ip}:30002?")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    finally:
        sock.close()

    return None


def set_digital_output(output_pin, value) -> bool:
    """
    controla las salidas digitales del robot
    """
    urscript_command = f"set_standard_digital_out({output_pin}, {'True' if value else 'False'})\n"

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((ip, 30002))
        sock.sendall(urscript_command.encode('utf-8'))
        return True
    
    finally:
        if 'sock' in locals():
            sock.close()



