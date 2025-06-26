import socket
import time

HOST = "192.168.0.100"  # IP del UR5
PORT = 63352        # Puerto del servidor URCap de Robotiq

def send_command(cmd):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall((cmd + '\n').encode())
        data = s.recv(1024)
    return data.decode().strip()

# Activar la pinza
#print("Activando gripper...")
#send_command("SET ACT 1")
#time.sleep(1)

# Establecer velocidad y fuerza
send_command("SET SPE 250")  # velocidad (0–255)
send_command("SET FOR 250")  # fuerza (0–255)
time.sleep(0.5)

spe = send_command("GET SPE")  # velocidad (0–255)
print("Posición actual:", spe)
fo = send_command("GET FOR")  # fuerza (0–255)
print("Fuerza actual:", fo)

# Cerrar la pinza (posición 255 = cerrado)
print("Cerrando gripper...")
send_command("SET POS 250")
time.sleep(2)

pos = send_command("GET POS")
print("Posición actual:", pos)

"""
# Obtener posición actual
pos = send_command("GET POS")
print("Posición actual:", pos)

# Abrir la pinza (posición 0 = abierto)
print("Abriendo gripper...")
send_command("SET POS 0")
time.sleep(2)

# Obtener posición actual
pos = send_command("GET POS")
print("Posición actual:",pos)

"""


def send_polyscope_pose(ip, port, pose_mm_rad, a=1.2, v=0.5):
    """
    pose_mm_rad = [x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]
    Envía un movej con get_inverse_kin a esa pose
    """
    x, y, z = [p / 1000.0 for p in pose_mm_rad[:3]]  # convertir mm a m
    rx, ry, rz = pose_mm_rad[3:]

    script = f"""
def move_to_pose():
    target_pose = p[{x}, {y}, {z}, {rx}, {ry}, {rz}]
    target_q = get_inverse_kin(target_pose)
    movej(target_q, a={a}, v={v})
end
move_to_pose()
"""

    print("Enviando URScript:\n", script)
    z = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    z.connect((ip, port))
    z.send(script.encode('utf-8'))
    time.sleep(0.5)
    z.close()



Toma_fotos = [-415, 351, 75, 3, -0.87, 0.016]

#"""
send_polyscope_pose(
    ip="192.168.0.100",
    port=30002,
    pose_mm_rad=Toma_fotos,
    a=1.2,
    v=0.5
)