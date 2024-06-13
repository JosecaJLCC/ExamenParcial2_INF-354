# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:49:08 2024

@author: joswe
"""

import numpy as np

# Definir el problema de la mochila
valores = np.array([60, 100, 120, 80, 50, 70])
pesos = np.array([10, 20, 30, 40, 50, 60])
capacidad = 100

# Generar una solución inicial aleatoria
def generar_solucion_inicial(n):
    solucion = np.random.randint(2, size=n)
    while np.dot(solucion, pesos) > capacidad:
        solucion = np.random.randint(2, size=n)
    return solucion

# Función de evaluación
def evaluar(solucion):
    valor_total = np.dot(solucion, valores)
    peso_total = np.dot(solucion, pesos)
    if peso_total <= capacidad:
        return valor_total
    else:
        return 0

# Vecindad basada en intercambio
def Swap(solucion):
    n = len(solucion)
    vecinos = []
    for i in range(n):
        vecino = solucion.copy()
        vecino[i] = 1 - vecino[i]  # Intercambiar 0 por 1 y viceversa
        if np.dot(vecino, pesos) <= capacidad:
            vecinos.append(vecino)
    return vecinos

# Vecindad basada en decremento
def Decrease(solucion):
    n = len(solucion)
    vecinos = []
    for i in range(1, n):
        vecino = solucion.copy()
        for j in range(n):
            vecino[j] = (solucion[j] - i) % 2  # Decremento en modulo 2
        if np.dot(vecino, pesos) <= capacidad:
            vecinos.append(vecino)
    return vecinos

# Solución basada en búsqueda de vecinos
def busqueda_vecinos(solucion_inicial, iteraciones=100):
    mejor_solucion = solucion_inicial
    mejor_valor = evaluar(mejor_solucion)
    
    for _ in range(iteraciones):
        vecinos = Swap(mejor_solucion) + Decrease(mejor_solucion)
        for vecino in vecinos:
            valor = evaluar(vecino)
            if valor > mejor_valor:
                mejor_solucion = vecino
                mejor_valor = valor
    return mejor_solucion, mejor_valor

# Generar una solución inicial
solucion_inicial = generar_solucion_inicial(len(valores))

# Ejecutar la búsqueda de vecinos
mejor_solucion, mejor_valor = busqueda_vecinos(solucion_inicial)

print(f'Solución inicial: {solucion_inicial}, Valor: {evaluar(solucion_inicial)}')
print(f'Mejor solución: {mejor_solucion}, Mejor valor: {mejor_valor}')