#!/usr/bin/env python3
import socket
import struct
# codigo sacado de https://forum.universal-robots.com/t/retrieving-information-about-digital-inputs-through-the-real-time-interface/1811
# se paso a python 3 y quito comentarios
# el codigo funciona para leer las intradas digitales y las salidas digitales


def decodeData(data, offset=0):
    i = offset

    # Longitud total del paquete
    packLength = struct.unpack_from(">i", data, i)[0]
    i += 4

    # Tipo de mensaje (normalmente 16)
    messageType = data[i]
    i += 1
    print(f"[INFO] Paquete largo: {packLength} bytes, tipo: {messageType}")

    if messageType == 16:
        while i < offset + packLength:
            subLength, subType = struct.unpack_from(">IB", data, i)
            if subType == 3:
                # subType 3 → paquete de tipo IO
                digital_input_bits, digital_output_bits = struct.unpack_from(">II", data, i + 5)
                print(f"[IO] Entradas (DI): {digital_input_bits:08b}, Salidas (DO): {digital_output_bits:08b}")

                # Puedes convertir a listas si lo prefieres:
                di = [(digital_input_bits >> bit) & 1 for bit in range(8)]
                do = [(digital_output_bits >> bit) & 1 for bit in range(8)]
                print(f"[IO] DI bits: {di}")
                print(f"[IO] DO bits: {do}")

            i += subLength

    return packLength


def main(ip="192.168.0.100", port=30002, iterations=100):
    print(f"[INFO] Conectando a {ip}:{port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))

    try:
        for _ in range(iterations):
            data = sock.recv(10000)
            if not data:
                continue

            total_received = len(data)
            interpreted = 0
            messages = 0

            while interpreted < total_received:
                interpreted += decodeData(data, interpreted)
                messages += 1

            print(f"[INFO] Paquete de {total_received} bytes, {messages} mensajes interpretados\n")

    finally:
        print("[INFO] Cerrando socket")
        sock.close()

if __name__ == "__main__":
    main()
