import random

# Definiujemy metaklasę
class GeneticMeta(type):
    def __new__(cls, name, bases, class_dict):
        # Parametry algorytmu genetycznego
        population_size = class_dict.get('population_size', 10)
        generations = class_dict.get('generations', 10)
        mutation_rate = class_dict.get('mutation_rate', 0.1)

        # Funkcja celu
        def fitness(individual):
            # Prosta funkcja celu - suma kwadratów atrybutów
            return sum(v ** 2 for v in individual.values())

        # Inicjalizacja populacji
        population = []
        for _ in range(population_size):
            individual = {k: random.uniform(-10, 10) for k in class_dict if not k.startswith('__')}
            population.append(individual)

        # Główna pętla algorytmu genetycznego
        for gen in range(generations):
            # Ocena populacji
            population = sorted(population, key=fitness, reverse=True)

            # Selekcja najlepszych
            next_generation = population[:population_size // 2]

            # Krzyżowanie (crossover)
            while len(next_generation) < population_size:
                parent1 = random.choice(next_generation)
                parent2 = random.choice(next_generation)
                child = {k: (parent1[k] + parent2[k]) / 2 for k in parent1}
                next_generation.append(child)

            # Mutacja
            for individual in next_generation:
                if random.random() < mutation_rate:
                    gene_to_mutate = random.choice(list(individual.keys()))
                    individual[gene_to_mutate] += random.uniform(-1, 1)

            population = next_generation

        # Przypisz najlepsze rozwiązanie do atrybutów klasy
        best_individual = population[0]
        for k, v in best_individual.items():
            class_dict[k] = v

        # Tworzenie klasy
        return super().__new__(cls, name, bases, class_dict)

# Klasa potomna wykorzystująca algorytm genetyczny
class OptimizedClass(metaclass=GeneticMeta):
    a = 5
    b = 10
    c = -3

    # Parametry algorytmu genetycznego
    population_size = 20
    generations = 50
    mutation_rate = 0.05

# Sprawdzenie atrybutów po optymalizacji
print(f"a: {OptimizedClass.a}")
print(f"b: {OptimizedClass.b}")
print(f"c: {OptimizedClass.c}")
