import math
import time
import random
from functools import reduce
from operator import add


def rad_to_deg(rad):
    return 180 / math.pi * rad


def deg_to_rad(deg):
    return math.pi / 180 * deg


def create_individual(length, min, max):
    """
    Creates an individual for a population
    :param length: the number of values in the list
    :param min: the minimum value in the list of values
    :param max: the maximal value in the list of values
    :return:
    """
    return [random.uniform(min, max) for x in range(length)]


def create_population(count, length, min_value, max_value):
    """
    Create a number of individuals (i.e., a population).
    :param count: the desired size of the population
    :param length: the number of values per individual
    :param min_value: the minimum in the individual’s values
    :param max_value: the maximal in the individual’s values
    """
    return [create_individual(length, min_value, max_value) for _ in range(count)]


def fitness(individual, target):
    """
    Determine the fitness of an individual. Lower is better.
    :param individual: the individual to evaluate
    :param target: the sum that we are aiming for (X)
    """
    L1 = 2  # meter
    L2 = 2  # meter
    L3 = 0.2  # meter
    Vx = 2  # m/s
    Vy = -1  # m/s
    By = 3  # meter

    s1 = individual[0]
    s2 = individual[1]
    s3 = individual[2]

    # fmt: off
    a = (L1 * math.cos(s1) + L2 * math.cos(s1 + s2) + L3 * math.cos(s1 + s2 - s3)) ** 2
    b = (By * -1 + L1 * math.sin(s1) + L2 * math.sin(s1 + s2) + L3 * math.sin(s1 + s2 - s3)) ** 2
    c = (L3 * -1 * Vx * math.cos(s1 + s2 - s3) + L3 * Vy * math.sin(s1 + s2 - s3)) ** 2
    # fmt: on

    return a + b + c


def grade(population, target):
    """
    Find average fitness for a population
    :param population: population to evaluate
    :param target: the value that we are aiming for (X)
    """
    total = sum([fitness(x, target) for x in population])
    return total / len(population)


def crossover(male, female):
    if random.random() > 0.5:
        return [male[0], male[1], female[2]]
    else:
        return [female[0], female[1], male[2]]


def mutate(x):
    idx = random.randint(0, 2)

    # Fully randomize a value (low chance)
    if random.random() > 0.90:
        x[idx] = random.uniform(0, math.pi)
    else:
        # Slightly increase/decrease a value
        change = random.uniform(0, math.pi) / 8
        if random.random() > 0.5:
            x[idx] += change
        else:
            x[idx] -= change


def evolve(population, target, retain=0.2, random_select=0.05, mutation_rate=0.01):
    """
    Function for evolving a population , that is, creating offspring (next generation population) from combining (crossover) the fittest individuals of the current population
    :param population: the current population
    :param target: the value that we are aiming for
    :param retain: the portion of the population that we allow to spawn offspring
    :param random_select: the portion of individuals that are selected at random, not based on their score
    :param mutation_rate: the amount of random change we apply to new offspring
    :return: next generation population
    """
    graded = [(fitness(x, target), x) for x in population]
    graded = [x[1] for x in sorted(graded, key=lambda x: x[0])]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]

    # Randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)

    # Crossover parents to create offspring
    desired_length = len(population) - len(parents)
    children = []

    while len(children) < desired_length:
        male = random.randint(0, len(parents) - 1)
        female = random.randint(0, len(parents) - 1)

        if male != female:
            children.append(crossover(parents[male], parents[female]))

    # Mutate some individuals
    for individual in children:
        if mutation_rate > random.random():
            mutate(individual)

    parents.extend(children)
    return parents


if __name__ == "__main__":
    start = time.time()
    target = 0
    population = create_population(100, 3, 0, math.pi)

    # Initial score
    score = grade(population, target)
    print("Average score: {}".format(score))

    for _ in range(100):
        population = evolve(population, target, 0.2, 0.1, 0.05)
        score = grade(population, target)
        print("Average score: {}".format(score))

    # Print the 5 best different solutions
    graded = [(fitness(x, target), x) for x in population]
    graded = sorted(graded, key=lambda x: x[0])

    duration = time.time() - start
    print("That took {:.2f} seconds".format(duration))

    last_score = None
    count = 0

    print("Printing best unique solutions (in deg)")
    for _, solution in enumerate(graded):
        if last_score != solution[0]:
            solution_in_deg = [rad_to_deg(x) for x in solution[1]]
            print("Score {:.6f} = {}".format(solution[0], solution_in_deg))
            count += 1
            last_score = solution[0]

            if count == 5:
                break
