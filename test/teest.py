import socket
import time

UR_IP = "192.168.0.100"
PORT = 30002  # Puerto de URScript

def send_script(script):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((UR_IP, PORT))
    s.send(script.encode('utf-8'))
    time.sleep(0.5)  # Asegura que URScript sea recibido antes del cierre
    s.close()

# Función URScript para mover y esperar hasta llegar
def movej_blocking(joints):
    return f"""
def move_and_wait():
  movej({joints}, a=1.2, v=0.5)
  while (not is_steady()):
    sleep(0.1)
  end
end

move_and_wait()
"""

# Enviar primer movimiento
send_script(movej_blocking("[1.77, -1.57, 1.57, -1.57, 1.57, 0.0]"))

# Espera en Python adicional por seguridad
time.sleep(1)

# Enviar segundo movimiento
send_script(movej_blocking("[1.17, -1.27, 1.17, -1.57, 1.57, 0.0]"))
