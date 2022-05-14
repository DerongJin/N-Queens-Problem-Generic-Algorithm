# --------------------
# @author: DR Jin
# @date: 2022.01.27
# --------------------

import random
import numpy as np
import time
import os
print('-----------------------------------------------------------------')
print("The performance of this GA is largely depends on the randomization,\nso it can not ensure to find a solution within time limit!!! \nIt's all depends on the luck.")
print('------------------------------------------------------------------')
print('------------------------------------------------------------------')
print('GOOD LUCK!!')
print('------------------------------------------------------------------')



# Parameter setting
print("Please type in mutation rate.")
mutationRate = float(input('0 < mutation rate < 1 (0.8 recommeded): '))    # mutation rate
print("---------------------------------")
print("Please type in population size.")
population = int(input('population size is an even number, less than 40 (recommended): '))      # size of populations
print("---------------------------------")
print("Please type in number of queens.")
numberOfQueens = int(input('number of queens is an interger: '))    # number of queens
print("---------------------------------")
# create random chromosome
def random_chromosome(size):
    chromosome = [i for i in range(1,size+1)]
    random.shuffle(chromosome)
    return chromosome

# generate random initial populations
def generate_random_populations(num,population):
    initialPopulation = []
    for i in range(population):
        initialPopulation.append(random_chromosome(num))
    return initialPopulation

# fitness function
# -----------------------
# Note: penalty = number of violated queens
#       fitness = chess board size - penalty (to be maximized)
def fitness_function(chromosome):
    penalty = 0
    for i in range(len(chromosome)):
        for j in range(i+1,len(chromosome)):
            if abs(i-j) == abs(chromosome[i] - chromosome[j]):
                penalty = penalty + 1
    fitness = (len(chromosome) * (len(chromosome) - 1))/2 - penalty
    return fitness

# crossover procedure
# -----------------------
# Note: return new population list
def crossover(matingPool, matePairs, crossPosition):
    newPopulation=[]
    for i in range(len(matingPool)):
        papa = matingPool[i]
        mama = matingPool[matePairs[i]-1]
        offspring = papa[:crossPosition[i]] + mama[crossPosition[i]:] + mama[:crossPosition[i]]
        offspring = list(reversed(offspring))
        for j in papa[:crossPosition[i]]:
            offspring.remove(j)
        offspring = list(reversed(offspring))
        newPopulation.append(offspring)
    return newPopulation

# generate mating pool
def generate_mating_pool(numOfSelection):
    matingPool=[]
    for i in range(len(numOfSelection)):
        if numOfSelection[i] != 0:
            matingPool = matingPool + [populations[i]]*numOfSelection[i]
    random.shuffle(matingPool)
    return matingPool

# randomly create mating pairs
def random_mating_list(num):
    L = [i for i in range(1,num+1)]
    interList = []
    while len(interList) != num:
        i = random.randint(0, num-1)
        while i in interList:
            i = random.randint(0, num-1)
        j = random.randint(0, num-1)
        while j == i or j in interList:
            j = random.randint(0, num-1)
        L[i], L[j] = L[j], L[i]
        interList.append(i)
        interList.append(j)
    return L

# generate random crossover position
def generate_crossover_position(numOfQueens, numOfPopulation):
    crossList = []
    for i in range(numOfPopulation):
        crossList = crossList + [random.randint(1,numOfQueens-1)]
    return crossList

# return sorted populations based on decresing fitness values
def sort_populations(populations):
    values = [fitness_function(i) for i in populations]
    values = np.array(values)
    newPopulations = [populations[i] for i in list(np.argsort(-values))]
    return newPopulations

# generate next populations based on offsprings
def generate_next_populations(offsprings, populations):
    offsprings = sort_populations(offsprings)
    populations = sort_populations(populations)

    for i in range(len(offsprings)):
        newFitness = fitness_function(offsprings[i])
        for j in range(len(populations)):
            if newFitness > fitness_function(populations[j]):
                populations[j] = offsprings[i]
                break

    return populations

# mutation
def mutatate(populations, mutationRate):
    if random.random() < mutationRate:

        mutatePopulationIndex = random.randint(0,len(populations) - 1)
        mutatePopulation = populations[mutatePopulationIndex]

        mutatePosition1, mutatePosition2 = random.randint(0,numberOfQueens-1), random.randint(0,numberOfQueens-1)
        while mutatePosition2 == mutatePosition1:
            mutatePosition2 = random.randint(0,numberOfQueens-1)
        mutatePopulation[mutatePosition1], mutatePopulation[mutatePosition2] = mutatePopulation[mutatePosition2], mutatePopulation[mutatePosition1]

        populations[mutatePopulationIndex] = mutatePopulation
    return populations


# check whether population has the correct n queens solutions
def check(population):
    for i in population:
        if fitness_function(i) == (numberOfQueens * (numberOfQueens - 1))/2:
            return True
    return False

# function to print the board
def print_result(population):
    for i in population:
        print('|-| '*(i-1) + '|O| ' + '|-| '*(numberOfQueens-i) )


if __name__ == '__main__':
    time_start=time.time()
    populations = generate_random_populations(numberOfQueens,population) # initial random population generation
    generation = 0
    while not check(populations): # termination condition (find the solution)
        fitness = [fitness_function(i) for i in populations] # fitness values of all identities in populations

        # selection criteria
        numOfTotal = [i / sum(fitness) for i in fitness]
        numOfSelection = [round(i*population) for i in numOfTotal]
        if sum(numOfSelection) != population:
            index = numOfSelection.index(max(numOfSelection))
            numOfSelection[index] = numOfSelection[index] + population - sum(numOfSelection)
        matingPool = generate_mating_pool(numOfSelection) # mating pool based on fitness values
        mateList = random_mating_list(population)
        crossoverPosition = generate_crossover_position(numberOfQueens,population)

        # offsprings generations (next generation configuration including mutation)
        offsprings = crossover(matingPool,mateList,crossoverPosition) # offsprings
        newOffsprings = mutatate(offsprings,mutationRate) # offsprings after mutation
        populations = generate_next_populations(newOffsprings,populations)
        generation = generation + 1

        if generation % 100 == 0:
            fitnessmax = max([fitness_function(i) for i in populations])
            print('Current generation:',generation, '---- Current fitness value:',fitnessmax, '---- Target value:', (numberOfQueens * (numberOfQueens - 1))/2)
    time_end=time.time()

    # print result chess board and final generation
    for i in populations:
        if fitness_function(i) == (numberOfQueens * (numberOfQueens - 1))/2:
            print('---------')
            print('Configurations: ',i)
            print('---------')
            print('Chess board visualization:')
            print_result(i)
    print('---------')
    print("Total generation: ",generation)
    print('Total time cost:', time_end-time_start, 'seconds')
print('---------')
print('Type in "Enter" to exit this program')
os.system("pause")

# some examples:
# 8 queens configuration:
# [5, 1, 8, 6, 3, 7, 2, 4]
#
# 50 queens configuration:
# [49, 13, 16, 9, 27, 29, 2, 18, 50, 28, 40, 37, 39, 22, 15, 25, 4, 10, 23, 14, 36, 1, 43, 41, 44, 17, 48, 34, 45, 42, 30, 5, 11, 24, 33, 20, 26, 35, 8, 47, 12, 6, 3, 46, 19, 21, 32, 7, 31, 38]
# [39, 27, 19, 35, 18, 28, 25, 20, 16, 11, 13, 42, 33, 50, 21, 14, 2, 47, 1, 31, 40, 46, 49, 7, 34, 30, 44, 38, 15, 5, 24, 10, 48, 4, 23, 26, 36, 22, 8, 43, 37, 6, 32, 9, 12, 17, 41, 29, 3, 45]
#
# 100 queens configuration:
# [66, 98, 16, 61, 65, 10, 89, 19, 92, 49, 86, 42, 75, 43, 74, 25, 44, 26, 45, 56, 4, 57, 60, 31, 12, 99, 50, 38, 27, 64, 1, 46, 5, 53, 8, 35, 85, 83, 80, 52, 72, 18, 6, 87, 30, 2, 21, 15, 37, 97, 73, 17, 95, 82, 71, 88, 54, 40, 51, 63, 29, 20, 22, 79, 9, 7, 24, 48, 90, 23, 32, 36, 78, 58, 70, 76, 3, 68, 96, 69, 14, 59, 77, 55, 33, 67, 41, 81, 28, 91, 13, 94, 84, 100, 47, 34, 93, 11, 39, 62]
# [69, 55, 31, 49, 35, 45, 97, 8, 84, 1, 90, 23, 33, 71, 58, 82, 43, 76, 86, 91, 22, 37, 79, 96, 66, 7, 50, 78, 25, 28, 68, 42, 16, 98, 48, 61, 73, 57, 3, 15, 95, 47, 94, 4, 24, 93, 40, 77, 27, 67, 13, 20, 74, 60, 26, 53, 5, 17, 62, 100, 6, 64, 87, 51, 83, 12, 41, 99, 19, 59, 70, 10, 30, 81, 89, 52, 85, 14, 11, 38, 54, 72, 36, 39, 80, 63, 29, 92, 56, 44, 21, 32, 75, 2, 18, 65, 46, 9, 34, 88]
#
# 200 queens configuration:
# [124, 153, 55, 144, 54, 113, 60, 82, 171, 18, 163, 125, 100, 128, 79, 135, 175, 68, 41, 158, 160, 182, 58, 189, 29, 50, 94, 69, 23, 110, 7, 101, 73, 142, 83, 90, 170, 197, 134, 26, 196, 169, 35, 5, 48, 103, 107, 84, 145, 22, 28, 127, 59, 96, 43, 89, 157, 45, 179, 136, 86, 143, 61, 53, 174, 77, 183, 133, 20, 148, 118, 111, 178, 49, 56, 1, 30, 150, 180, 85, 4, 159, 51, 8, 24, 185, 121, 186, 67, 193, 149, 74, 6, 63, 184, 152, 32, 166, 120, 15, 81, 36, 165, 12, 131, 64, 122, 126, 188, 200, 75, 57, 46, 151, 117, 87, 130, 164, 33, 137, 70, 38, 168, 194, 2, 80, 9, 199, 190, 11, 78, 146, 93, 129, 52, 109, 138, 37, 104, 147, 115, 172, 162, 187, 72, 16, 25, 192, 112, 116, 17, 19, 71, 39, 65, 155, 167, 95, 99, 102, 154, 114, 191, 195, 161, 31, 123, 177, 21, 132, 14, 98, 76, 47, 97, 105, 156, 34, 176, 66, 44, 13, 40, 139, 92, 198, 106, 173, 119, 181, 42, 10, 62, 140, 27, 3, 91, 88, 108, 141]


