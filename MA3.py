""" MA3.py

Student: Anna Larsson
Mail: anna.larsson.5255@student.uu.se
Reviewed by: 
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
import functools
from numba import njit

# Exc1
def approximate_pi(n):
    # n is the number of points
    circle_points = 0                                               # nc
    inside_x = []                                                   # skapar listor för att kunna spara värden för mina punkter
    inside_y = []
    outside_x = []
    outside_y = []

    for i in range(n):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)         # genererar slumpmässig koordinat för varje prick mellan -1, 1
        if (x**2 + y**2) <= 1:                                      # om pricken hamnar inom cirkeln
            circle_points += 1      
            inside_x.append(x)                                      # lägg till i listorna för i cirklen
            inside_y.append(y)
        else:
            outside_x.append(x)                                     # lägg till i listorna för utanför
            outside_y.append(y)                            
    tot_points = n                                            
    pi = 4* (circle_points / tot_points)                            # pi = 4 * antal punkter i cirkel / totala antal punkter

    print(pi)

    # plt.scatter(inside_x, inside_y, color="red")                    
    # plt.scatter(outside_x, outside_y, color="blue")
    # plt.savefig(f"pi_simulation_{n}.png")
    # plt.show()
    
    return pi


# Exc2, approximation
def sphere_volume(n, d):                # n is the number of points ,  d is the number of dimensions of the sphere 
# to do : förklara vilka koncept/funktioner som används och VARFÖR. 
    inside_sphere = 0
    outside_sphere = 0

    # för alla n gånger, skapa en punkt med d koordinater mellan -1 och 1
    for i in range(n):
        point = [random.uniform(-1, 1) for j in range(d)]       # list comprehension

    # kvadrera koordinaterna med map + lambda
        squared = list(map(lambda x: x**2, point))              # skapar en lista med våra datapunkter fast kvadrerade. Utan list skapar vi ett map-objekt 
                                                                # lambda läses : ' för x, ta x**2'
    
    # summera alla kvadrater med reduce
        sum_of_squares = functools.reduce(lambda x, y: x + y, squared)        # förklara hur reduce fungerar
    
    # testar villkoret för om punkten ligger i hypersfären.
        if sum_of_squares <= 1:
            inside_sphere += 1
        else:
            outside_sphere += 1
                                            
    volume = (2**d) * (inside_sphere / n)                       # inside_sphere / n ger andelen hyperkubder i hypersfären. 2^d ger volym för en hyperkub
    return volume


#Exc2, real value
def hypersphere_exact(n, d):    # n is the number of points, d is the number of dimensions of the sphere 
    volume = (m.pi**(d/2)) / (m.gamma(d/2+1))
    return volume


#Exc3: numba version
# När vi kör koden tre gånger med vanlig python tar det ca 2.16 - 2.19 sekunder för n = 10^6 och d = 11
# När vi kör koden tre gånger med numba tar det ca 0.3 - 0.5 sekunder. 
# Första gången måste Numba först kompilera funktionen till maskinkod. andra + gången kan numba använda samma kompilerade verision

@njit                                                              # när funktionen körs, kompilera den med Numba
def sphere_volume_numba(n:int, d:int)->float:                      # n: number of points,  d: number of dimensions of the sphere
    inside_sphere = 0
    for i in range(n):                                             # för varje n
        point = []
        for j in range(d):                                         # skapa en punkt med d random koordinater
            point.append(random.uniform(-1, 1))
    
        sum_of_squares = 0                              
        for k in point:                                            
            sum_of_squares = sum_of_squares + k**2                 # summera punktenens koordinater i kvadrat
    
        if sum_of_squares <= 1:                                    # avgör om just denna punkt ligger i sfären 
            inside_sphere += 1                                     # addera isåfall +1 till räknaren 

    volume = (2**d) * (inside_sphere / n)
    return volume


#Exc4: parallel code - parallelize actual computations by splitting data
# mha parallelisering kan vi bearbeta data parallellt samtidigt. CPU = Central Processing Unit
def sphere_volume_parallel(n, d, np=10):     # np = number of processes
    points_per_process = n // np                                   # delar det totala antalet punkter mellan processerna 
    "skapa en grupp workers som kan ta sphere_volume-beräkningar "

    with future.ProcessPoolExecutor(max_workers=np) as ex:         # skapa en processpool med högst np workers. ex är namnet på processpool
        results = ex.map(sphere_volume,   [points_per_process for _ in range(np)],  [d for _ in range(np)])
        # Argument : 1 - funktionen som varje worker ska köra, 2 - antal punkter för sphere_volume , 3 - antal dimensioner

    return mean(results)                                           # returnerar medelvärdet av alla parallella resultat
    

    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    # for n in dots:
    #     approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    
    # print(f"Approximated volume of {d} dimentional sphere = {sphere_volume(n,d)}")
    # print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    
    # print(f"Approximated volume of {d} dimentional sphere = {sphere_volume(n,d)}")
    # print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    for i in range(3):
        start = pc()
        sphere_volume(n, d)
        stop = pc()
        print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
   
    for i in range(3):
        start = pc()
        sphere_volume_numba(n, d)
        stop = pc()
        print(f"Exc3 : Numba time: {stop-start}")

    # Exc4
    n = 1000000
    d = 11
    for i in range(3):
        start = pc()
        sphere_volume(n, d)
        stop = pc()
        # print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")

    for i in range(3):
        start = pc()
        sphere_volume_parallel(n, d)
        stop = pc()
        print(f"Exc4 : Parallell time: {stop-start}")

    
    

if __name__ == '__main__':
	main()


