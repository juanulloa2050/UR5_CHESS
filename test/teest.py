import socket
import time

def set_digital_output_script(ip: str, output_pin: int, value, script_port: int = 30002) -> bool:
    """
    Sets a standard digital output on the UR robot by sending a URScript command
    to the script interface (port 30002).
    """
    if not (0 <= output_pin <= 7):
        print(f"[ERROR] Digital output pin {output_pin} is out of valid range (0-7).")
        return False

    if not isinstance(value, bool):
        print(f"[ERROR] 'value' must be a boolean (True or False), not {type(value).__name__}.")
        return False

    # Must use "True" or "False" as string literals in URScript
    urscript_command = f"set_standard_digital_out({output_pin}, {'True' if value else 'False'})\n"

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((ip, script_port))
        sock.sendall(urscript_command.encode('utf-8'))
        return True

    except socket.timeout:
        print(f"[ERROR] Timeout while connecting or sending command to {ip}:{script_port}.")
        return False
    except ConnectionRefusedError:
        print(f"[ERROR] Connection refused. Is the robot in remote control mode?")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False
    finally:
        if 'sock' in locals():
            sock.close()


# --- Ejemplo de uso ---
if __name__ == "__main__":
    robot_ip = "192.168.0.100"
    digital_output_pin = 0

    print(f"\n--- Activando salida digital {digital_output_pin} ---")
    if set_digital_output_script(robot_ip, digital_output_pin, True):
        print("✔️ Salida activada correctamente.")
    else:
        print("❌ Error al activar la salida.")

    time.sleep(3)

    print(f"\n--- Desactivando salida digital {digital_output_pin} ---")
    if set_digital_output_script(robot_ip, digital_output_pin, False):
        print("✔️ Salida desactivada correctamente.")
    else:
        print("❌ Error al desactivar la salida.")
