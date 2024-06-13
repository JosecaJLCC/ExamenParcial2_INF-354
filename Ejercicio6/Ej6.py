# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:59:41 2024

@author: joswe
"""

import numpy as np
import random

# Funciones de vecindad
def TwoOpt(perm):
    n = len(perm)
    n_neighbors = int(n*(n-1)/2 - n)
    neighbors = []
    for i in range(n-1):
        for j in range(i+2, n):
            if not (i == 0 and j == n-1):
                neighbor = perm[:]
                neighbor[i:j+1] = reversed(perm[i:j+1])
                neighbors.append(neighbor)
    return neighbors

def Swap(perm):
    n = len(perm)
    neighbors = []
    for i in range(1, n):
        neighbor = perm[:]
        neighbor[i], neighbor[0] = neighbor[0], neighbor[i]
        neighbors.append(neighbor)
    return neighbors

def Complement(perm):
    n = len(perm)
    neighbor = [(n - x) if x != n else n for x in perm]
    return [neighbor]

def Decrease(perm):
    n = len(perm)
    neighbors = []
    for i in range(1, n):
        neighbor = [(x - i) if (x - i) > 0 else (x - i + n) for x in perm]
        neighbors.append(neighbor)
    return neighbors

# Función para calcular la distancia total de una ruta
def calculate_distance(route, distance_matrix):
    distance = 0
    for i in range(len(route)):
        distance += distance_matrix[route[i-1]][route[i]]
    return distance

# Función para generar una ruta aleatoria
def random_route(num_cities):
    route = list(range(num_cities))
    random.shuffle(route)
    return route

# Función para realizar la selección por torneo
def tournament_selection(population, distance_matrix, tournament_size):
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda x: calculate_distance(x, distance_matrix))
    return tournament[0]

# Función para realizar el cruce de dos rutas
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end+1] = parent1[start:end+1]
    pointer = 0
    for city in parent2:
        if city not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = city
    return child

# Función para realizar la mutación de una ruta
def mutate(route, mutation_rate):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route)-1)
            route[i], route[j] = route[j], route[i]

# Función principal del algoritmo genético
def genetic_algorithm(distance_matrix, population_size=100, generations=500, mutation_rate=0.01, tournament_size=5, neighborhood_func=TwoOpt):
    num_cities = len(distance_matrix)
    population = [random_route(num_cities) for _ in range(population_size)]

    for generation in range(generations):
        new_population = []
        for _ in range(population_size):
            parent1 = tournament_selection(population, distance_matrix, tournament_size)
            parent2 = tournament_selection(population, distance_matrix, tournament_size)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population

        # Evaluar la mejor ruta de la generación actual
        best_route = min(population, key=lambda x: calculate_distance(x, distance_matrix))
        best_distance = calculate_distance(best_route, distance_matrix)
        # print(f'Generación {generation+1}: Mejor distancia = {best_distance}')

        # Aplicar vecindad
        if neighborhood_func:
            neighbors = neighborhood_func(best_route)
            for neighbor in neighbors:
                neighbor_distance = calculate_distance(neighbor, distance_matrix)
                if neighbor_distance < best_distance:
                    best_route = neighbor
                    best_distance = neighbor_distance

    best_route = min(population, key=lambda x: calculate_distance(x, distance_matrix))
    return best_route, calculate_distance(best_route, distance_matrix)

# Función para convertir una ruta de índices a letras
def indices_to_letters(route):
    letters = ['A', 'B', 'C', 'D', 'E']
    return [letters[i] for i in route]

# Ejemplo de uso
if __name__ == "__main__":
    # Matriz de distancias de ejemplo (grafo completo)
    distance_matrix = np.array([
        [0,   7,  9,  8, 20],
        [7,   0, 10,  4, 11],
        [9,  10,  0, 15,  5],
        [8,   4, 15,  0, 17],
        [20, 11,  5, 17,  0],
    ])

    # Ejecutar el algoritmo genético
    best_route, best_distance = genetic_algorithm(distance_matrix, neighborhood_func=TwoOpt)
    best_route_letters = indices_to_letters(best_route)
    print()
    print(f'Mejor ruta encontrada: {best_route_letters}')
    print(f'Distancia de la mejor ruta: {best_distance}')