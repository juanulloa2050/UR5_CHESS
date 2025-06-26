import numpy as np
import math

p_h8 = np.array([-420.40, 26.45])   # Casilla H8 (origen local del tablero)
p_a1 = np.array([-217.45, 422.20])  # Casilla A1 (opuesta en diagonal)
p_a8 = np.array([-225.16, 421.49])   # Casilla A8

z = -45  # Altura de referencia (mm)

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

# Punto en el sistema local
p_local = np.array([440.65, 0, 0, 1])

# Punto transformado al sistema global
p_global = T @ p_local
print("Coordenadas globales:", p_global)
