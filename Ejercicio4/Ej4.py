# -*- coding: utf-8 -*-
"""
@author: joswe
"""
"""RESPUESTA:

El algoritmo de Recocido Simulado (Simulated Annealing) es una técnica de 
optimización metaheurística que puede aplicarse efectivamente a problemas donde
no existe una solución efectiva conocida o donde es difícil encontrar la solución
óptima debido a la complejidad del espacio de búsqueda

Ejemplo: Aplicación de Recocido Simulado al Problema del Agente Viajero (TSP)
El Problema del Agente Viajero (TSP) implica encontrar la ruta más corta que visita cada ciudad exactamente una vez y regresa al punto de partida.

"""

import numpy as np

# Definir matriz de distancias entre ciudades (ejemplo de 5 ciudades)
dist_matrix = np.array([
    [0, 7, 9, 8, 20],   # Distancias desde la ciudad A a las demás ciudades
    [7, 0, 10, 4, 11],  # Distancias desde la ciudad B a las demás ciudades
    [9, 10, 0, 15, 5],  # Distancias desde la ciudad C a las demás ciudades
    [8, 4, 15, 0, 17],  # Distancias desde la ciudad D a las demás ciudades
    [20, 11, 5, 17, 0]  # Distancias desde la ciudad E a las demás ciudades
])

# Número de ciudades
n_cities = dist_matrix.shape[0]



"""
Evalúa la función objetivo del TSP para una solución dada.
Argumentos:
- solution (list): Solución del TSP (orden de las ciudades).
- dist_matrix (ndarray): Matriz de distancias entre ciudades.
Retorna:
- float: El costo total del recorrido.
"""
# Función para evaluar la función objetivo del TSP para una solución dada
def evaluate_tsp_solution(solution, dist_matrix):
    total_distance = 0
    for i in range(len(solution)):
        from_city = solution[i]
        to_city = solution[(i + 1) % len(solution)]
        total_distance += dist_matrix[from_city - 1, to_city - 1]  # Ajuste para índices de 1 a n
    return total_distance



# Implementación de Recocido Simulado para TSP
def simulated_annealing_tsp(dist_matrix, initial_solution, initial_temperature=1000, cooling_rate=0.95, stopping_temperature=1e-6):
    current_solution = initial_solution[:]
    best_solution = current_solution[:]
    current_cost = evaluate_tsp_solution(current_solution, dist_matrix)
    best_cost = current_cost
    
    temperature = initial_temperature
    
    while temperature > stopping_temperature:
        # Generar un vecino aleatorio intercambiando dos ciudades
        new_solution = current_solution[:]
        swap_index1 = np.random.randint(0, len(new_solution))
        swap_index2 = np.random.randint(0, len(new_solution))
        new_solution[swap_index1], new_solution[swap_index2] = new_solution[swap_index2], new_solution[swap_index1]
        
        # Calcular el costo del nuevo vecino
        new_cost = evaluate_tsp_solution(new_solution, dist_matrix)
        
        # Decidir si aceptar el nuevo vecino basado en la función de probabilidad de Boltzmann
        delta_cost = new_cost - current_cost
        acceptance_probability = np.exp(-delta_cost / temperature)
        
        if delta_cost < 0 or np.random.rand() < acceptance_probability:
            current_solution = new_solution
            current_cost = new_cost
        
        # Actualizar la mejor solución encontrada
        if current_cost < best_cost:
            best_solution = current_solution[:]
            best_cost = current_cost
        
        # Enfriamiento según el esquema seleccionado
        temperature *= cooling_rate
    
    return best_solution, best_cost




# Solución inicial aleatoria
initial_solution = np.random.permutation(np.arange(1, n_cities + 1))  # Permutación de [1, 2, 3, 4, 5]

# Ejecutar Recocido Simulado
best_solution, best_cost = simulated_annealing_tsp(dist_matrix, initial_solution)

# Mostrar resultados
print(f"Mejor solución encontrada: {best_solution}")
print(f"Costo de la mejor solución: {best_cost}")