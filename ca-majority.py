# This code simulates elementary CAs
# It prints a space-time plot of the CA behaviour

import numpy as np
import matplotlib.pyplot as plt
import random as rndm
import time
import datetime

##############################################################################
#                      Global variables
##############################################################################

ncells = 49                                                                     # Number of cells
ntsteps = 50                                                                    # Number of time steps
nic = 10                                                                        # number of initial conditions
psize = 10                                                                      # size of a population
ngenerations = 10                                                               # number of generations
density = 0                                                                   # density of the majority of the initial condition

##############################################################################
#                      Methods and functions
##############################################################################

def BinaryToDecimal(array):
    """
    Returns the decimal number for the input array that represents a rule.
    """
    res = 0
    for i in range(len(array)):
        res += array[i] * 2**i
    return(int(res))
    
def DecimalToBinary(getal):
    """
    Takes a decimal number and returns the appropiate binary nextstate.
    """
    getal = int(getal)
    rule = bin(getal)[::-1][0:-2]
    array = np.zeros(32)
    for i in range(len(rule)):
        array[i] = rule[i]
    return(array)

def RuleGenerator():
    """
    Creates a random nextstate rule of length 32.
    """
    state = int(np.random.uniform(0 , 2**32-1))
    state = DecimalToBinary(state)
    return(state)

def Majority(init_states):
    """
    Calculates what state was the majority in the initial state.
    """
    res = 0
    for i in init_states:
        res += i
    res = res / ncells
    if res > 0.5:
        res = 1
    else:
        res = 0
    return(res)
        
def FitnessScore(init_states,states):
    """
    Classifies if the nextstate function got the majority classification correct.
    """
    states = states[-1,:]                                                       # select the last row as the final answer of the CA
    res = 0
    for i in states:
        res += i                                                                # sum all of its cells
    if res == ncells:
        res = 1                                                                 # if all last cells are black it voted 1
    elif res == 0:
        res = 0                                                                 # if all last cells are white it voted 0
    else:   
        return(0)                                                               # didnt give a clear answer, so got score 0.
        
    correct = Majority(init_states)
    if res == correct:
        return(1)
    else:
        return(0)
        

    
def TimeLoop(states,buren):
    """
    Applies the next state function for a globally given amount of timesteps.
    Does this by looking up the 5-neighborhood in a dictionary (buren).
    """   
    for i in range(1,ntsteps):
        inputtime = i-1
        for j in range(0,ncells):
            buur1 = states[inputtime,(j-2) % ncells]
            buur2 = states[inputtime,(j-1) % ncells]
            cellf = states[inputtime,(j) % ncells]
            buur3 = states[inputtime,(j+1) % ncells]
            buur4 = states[inputtime,(j+2) % ncells]
            states[i,j] = buren[str(buur1)+str(buur2)+str(cellf)+str(buur3)+str(buur4)]
            

def PlotCA(states,nsnumber):
    """
    Plots the nextstate rule with the given init_states.
    """
    plt.matshow(states, cmap = 'Greys')
    plt.title("Nextstate function %d" % nsnumber)
    plt.ylabel("Time", fontsize = 24)
    plt.xlabel("Space", fontsize = 24)
    plt.yticks(fontsize = 20)
    plt.gca().xaxis.tick_bottom()
    plt.xticks(fontsize = 20)
    plt.show()

def TestIndividual(nextstate,plot = False):
    """
    Tests the given CA for the specified number of initial conditions.
    Set plot to True to plot the test cases and analyse the individual.
    """
    buren_x = []
    buren_y = []
    for i in range(32):
        nei = bin(i)[2:].zfill(5)
        buren_x.append(nei)
        buren_y.append(nextstate[i])
    
    buren = dict(zip(buren_x,buren_y))                                           # makes a dictionary to look up the nextstate value that is associated with the calculated neighbourhood.
    fitness = 0
    nsnumber = BinaryToDecimal(nextstate)
    states = np.zeros((ntsteps, ncells),dtype=int)      
    for i in range(nic):
        m = np.random.randint(2)                                                # chooses a random majority
        d = density
        states[0,:] = np.random.choice([abs(1-m), 1-abs(1-m)],ncells, p=[d,1-d])       # distributes this majority randomly with the given density.
        TimeLoop(states,buren)
        fitness += FitnessScore(states[0,:],states)
        if plot:
            PlotCA(states,nsnumber)
            if FitnessScore(states[0,:],states)==1:
                print(Majority(states[0,:]))
                print("correct")
            if FitnessScore(states[0,:],states)==0:
                print(Majority(states[0,:]))
                print("wrong")
    fitness = fitness / nic
    return([int(nsnumber),fitness])
    

def InitialPopulation(individuals,initial_conditions):
    """
    Makes a population of 50 random individuals and returns them with their fitness of 100 initial conditions.
    """
    population = []
    for i in range(individuals): 
        nextstate = RuleGenerator()
        I = TestIndividual(nextstate)
        population.append(I)
    return(population)
    
    
def Sort(lijst): 
    """
    Bubble sort, to sort individuals on their fitness score.
    """
    l = len(lijst) 
    for i in range(0, l): 
        for j in range(0, l-i-1): 
            if (lijst[j][1] > lijst[j + 1][1]): 
                hoger = lijst[j] 
                lijst[j]= lijst[j + 1] 
                lijst[j + 1] = hoger 
    return(lijst) 
    

def Selection(population):
    """
    Selects and returns 20% fittest individuals
    """
    population_size = len(population)
    s = int(0.2*population_size)                                                # calculates the size of the survivors
    survivors = population[-s::]                                                # selects the last 20%, since the input population will be sorted.
    selected_individuals = []                                                   # create a list to append the fittest individuals.
    
    for i in range(s):
        DNA = survivors[i][0]
        selected_individuals.append(DNA)
    return(selected_individuals,population)
    
    
def Generation(survivors,population,population_size):
    """
    Takes survivors and returns them with their offspring to fill a new population.
    """
    fitness = [individual[1] for individual in population]
    fitness_normalized = [float(i)/sum(fitness) for i in fitness]
    new_population = survivors
    survivor_size = len(survivors)
    
    for i in range(population_size-survivor_size):


#        p1_DNA = DecimalToBinary(survivors[p1])
        
        p1 = np.random.choice([i[0] for i in population],1, p=fitness_normalized)
        p1_DNA = DecimalToBinary(p1)
        p2 = np.random.choice([i[0] for i in population],1, p=fitness_normalized)
        p2_DNA = DecimalToBinary(p2)
        

        r_point = int(32*rndm.random())                                         # select a random point to recombine their DNA
        
        child = []
        
        p1_DNA = p1_DNA[0:r_point+1]
        p2_DNA = p2_DNA[r_point:-1]
        
        for base in p1_DNA:
            child.append(base)
        for base in p2_DNA:
            child.append(base)
        for i in range(32):
            mut = np.random.choice([1,0],1, p=[1/32,31/32])                     # every bit in its DNA has 1/32 chance of mutating.
            if mut==1:
                child[i] = abs(child[i]-1)
            
        child = BinaryToDecimal(child)
        new_population.append(int(child))
    return(new_population)
    
    
def LangtonLambda(nextstate):
    """
    Returns the Langton's lambda of the nextstate.
    """
    nextstate = DecimalToBinary(nextstate)
    s = 0
    for i in nextstate:
        s += i
    l = s / len(nextstate)
    return(l)
    

def TestPopulation(population):
    """
    Tests a population by taking an initial population.
    """
    result = []
    for individual in population: 
        individual = DecimalToBinary(individual)
        I = TestIndividual(individual)
        result.append(I)
    result = Sort(result)
    return(result)
    
    
def Evolve(p=0):
    lineage = []
    p = Sort(InitialPopulation(psize,nic))
    lineage.append(p)
    print("Initial population after testing fitness: \n")
    print(p)
    print("\n")
    p = Selection(p)
    p2=Generation(p[0],p[1],psize)
    res=p2
    for i in range(ngenerations-1):
        res = TestPopulation(res)
        lineage.append(res)
        print("Testresults of generation %d" %(i+2))
        print("\n")
        print(res)
        print("\n")
        res = Selection(res)
        res = Generation(res[0],res[1],psize)
        
    return(lineage)

def EvolveFurther(population):
    lineage = []
    p = population
    lineage.append(p)
    print("Initial population after testing fitness: \n")
    print(p)
    print("\n")
    p = Selection(p)
    p2=Generation(p[0],p[1],psize)
    res=p2
    for i in range(ngenerations-1):
        res = TestPopulation(res)
        lineage.append(res)
        print("Testresults of generation %d" %(i+2))
        print("\n")
        print(res)
        print("\n")
        res = Selection(res)
        res = Generation(res[0],res[1],psize)
        
    return(lineage)


##############################################################################
#                   Running the genetic algorithm
##############################################################################

#lineage = Evolve()

##############################################################################
#                   Saving the data
##############################################################################

#ts = time.time()
#tijd = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H-%M-%S')
#
#text_file = open("Output "+tijd+".txt", "w")
#text_file.write(str(lineage))
#text_file.close()

##############################################################################
#                   Reading a dataset
##############################################################################

file = open("lineage.txt", "r")
lineage = file.readline()
import re
a=re.split(', | \[ | \]',lineage[1:-1])

for i in range(len(a)):
    if i%2==0:
        a[i]=a[i][1::]
    else:
        if i!=len(a)-1:
            a[i]=a[i][:-1]
        
for i in range(len(a)):
    if i % 99 == 0:
        a[i] = a[i][0:-2]
    if i % 100 == 99:
        if i!=len(a)-1:
            a[i] = a[i][0:-1]
    if i % 100 == 0:
        a[i]=a[i][1:]

for i in range(len(a)):
    a[i] = float(a[i])


fitness = []
individuals = []
for i in range(len(a)):
    if (i + 1) % 2 == 0:
        fitness.append(a[i])
        individuals.append(a[i-1])
fittest = []
fittest_individuals = []

for i in range(len(fitness)):
    if (i+1)%50==0:
        fittest.append(fitness[i])
        fittest_individuals.append(individuals[i])
        
fittest[0]=fitness[48] 
print("Fitness over time: \n")      
print(fittest)
print("Fittest indivduals over the generations: \n")
print(fittest_individuals)


fit = np.polyfit(range(100),fittest,10)
poly = np.poly1d(fit) 
print(poly(90))
gens = range(0,100)
plt.scatter(gens,fittest,s=10)
plt.plot(range(95),poly(range(95)),c='black')
plt.ylim(0,1)
plt.xlabel("Generation")
plt.ylabel("Fitness (correct answer percentage)")
plt.show()




##############################################################################
#                   Evolving a dataset further
##############################################################################

a = a[-100:]
b=[]
for i in range(0,100,2):
    b.append([a[i],a[i+1]])
print("Last generation, with their results: \n")
print(b)

#lineage = EvolveFurther(b)

##############################################################################
#                   Analysing an individual
##############################################################################

test = TestIndividual(DecimalToBinary(4219502720),True)
print(test)

##############################################################################
#                   Saving the data
##############################################################################

#ts = time.time()
#tijd = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H-%M-%S')
#
#text_file = open("Output "+tijd+".txt", "w")
#text_file.write(str(lineage))
#text_file.close()
    
##############################################################################
#                   Getting the nextstate function of an individual
##############################################################################

#c = DecimalToBinary(4079526048)
#buren_x = []
#buren_y = []
#for i in range(32):
#    nei = bin(i)[2:].zfill(5)
#    buren_x.append(nei)
#    buren_y.append(c[i])
#    
#buren = dict(zip(buren_x,buren_y))
#
#print(buren)

##############################################################################
#                   Langton lambda analysis
##############################################################################
    
#l = [LangtonLambda(i) for i in fittest_individuals]
#import matplotlib.patches as patches
#
#fig,ax = plt.subplots(1)
#plt.plot(range(100),l)
#rect1 = patches.Rectangle((0,0),100,0.35,linewidth=1,edgecolor='springgreen',facecolor='lightgreen',label='Class II (periodic)')
#rect2= patches.Rectangle((0,0.35),100,0.2,linewidth=1,edgecolor='plum',facecolor='lightblue',label='Class IV (complex)')
#rect3 = patches.Rectangle((0,0.45),100,0.3,linewidth=1,edgecolor='salmon',facecolor='lightsalmon',label='Class III (chaos)')
#plt.ylim(0,0.7)
#ax.add_patch(rect1)
#ax.add_patch(rect2)
#ax.add_patch(rect3)
#plt.title("Langton's lambda for the fittest individual")
#plt.xlabel("Generation")
#plt.ylabel(r'$\lambda$')
#plt.legend(handles=[rect1,rect2,rect3],loc='upper right')
#plt.show()


    