from time import strptime
from load_flight_data import destination, people, load_flight_data, printschedule
from random import randint, random, choice
from math import e
from collections import deque


def getminutes(t):
    x = strptime(t, '%H:%M')
    return x[3] * 60 + x[4]


def schedulecost(sol, flights):
    totalprice = 0
    latestarrival = 0
    earliestdep = 24 * 60
    totalwait = 0

    for d in range(len(sol) // 2):
        origin = people[d][1]
        outbound = flights[(origin, destination)][int(sol[2 * d])]
        returnf = flights[(origin, destination)][int(sol[2 * d + 1])]

        totalprice += (outbound[2] + returnf[2])

        outbound_minutes = getminutes(outbound[1])
        if latestarrival < outbound_minutes:
            latestarrival = outbound_minutes

        returnf_minutes = getminutes(returnf[0])
        if earliestdep > returnf_minutes:
            earliestdep = returnf_minutes

    totalwait = 0

    for d in range(len(sol) // 2):
        origin = people[d][1]
        outbound = flights[(origin, destination)][int(sol[2 * d])]
        returnf = flights[(origin, destination)][int(sol[2 * d + 1])]

        totalwait += latestarrival - getminutes(outbound[1])
        totalwait += getminutes(returnf[0]) - earliestdep

    if latestarrival < earliestdep:
        totalwait += 50

    return totalprice + totalwait


def randomoptimization(domain, costf):
    best = 9999999999999
    bestr = None
    for i in range(1000):
        r = [randint(ll, ul) for ll, ul in domain]
        cost = costf(r)
        if cost < best:
            best, bestr = cost, r
    return bestr


def hillclimb(domain, costf, initial_guess=None):
    ''' also known as gradient descent '''
    if initial_guess is None:
        sol = [randint(ll, ul) for ll, ul in domain]
    else:
        sol = initial_guess
    while 1:
        neighbors = []
        for j, (ll, ul) in enumerate(domain):
            if sol[j] > ll:
                neighbor = sol[:]
                neighbor[j] -= 1
                neighbors.append(neighbor)
            if sol[j] < ul:
                neighbor = sol[:]
                neighbor[j] += 1
                neighbors.append(neighbor)

        current = costf(sol)
        best = current
        for neighbor in neighbors:
            cost = costf(neighbor)
            if cost < best:
                best, sol = cost, neighbor

        if best == current:
            return sol


def annealing(domain, costf, initial_guess=None, T=10000.0, cool=0.99, step=1):
    if initial_guess is None:
        sol = [randint(ll, ul) for ll, ul in domain]
    else:
        sol = initial_guess

    current_cost = costf(sol)
    while T > 0.1:
        i = randint(0, len(domain) - 1)
        dir_ = step * choice([-1, 1])
        vecb = sol[:]
        vecb[i] += dir_
        ll, ul = domain[i]
        if vecb[i] < ll:
            vecb[i] = ll
        if vecb[i] > ul:
            vecb[i] = ul

        costb = costf(vecb)

        if costb < current_cost:
            sol, current_cost = vecb, costb

        else:
            p = pow(e, (-costb - current_cost) / T)
            if random() < p:
                sol = vecb
        T *= cool
    return sol


def annealing_grid(domain, costf, T=10000.0, cool=0.99, step=1, pop=10):
    initial_guesses = [[randint(ll, ul) for ll, ul in domain]
                       for _ in range(pop)]
    solns = [annealing(domain, costf, initial_guess=initial_guess,
                       T=T, cool=cool, step=step)
             for initial_guess in initial_guesses]

    scores = [(soln, costf(soln)) for soln in solns]
    scores.sort(key=lambda x: x[1])
    return scores[0][0]


def genetic(domian, costf, popsize=50, step=1,
            mutprob=0.2, elite=0.2, maxiter=200, no_improve_break=None):
    if no_improve_break is not None:
        last_scores = deque([], maxlen=no_improve_break)

    def mutate(vec):
        mutated = vec[:]
        i = randint(0, len(domain) - 1)
        ll, ul = domain[i]
        if (random() < 0.5) and (vec[i] > ll):
            mutated[i] -= step
        elif vec[i] < ul:
            mutated[i] += step
        return mutated

    def crossover(r1, r2):
        i = randint(0, len(domain) - 1)
        return r1[:i] + r2[i:]

    pop = [[randint(ll, ul) for ll, ul in domain] for _ in range(popsize)]
    topelite = int(elite * popsize)
    for generation in range(maxiter):
        scores = [(costf(v), v) for v in pop]
        scores.sort()
        ranked = [v for s, v in scores]
        pop = ranked[:topelite]
        while len(pop) < popsize:
            breeders = ranked[:topelite]
            if random() < mutprob:
                c = choice(breeders)
                pop.append(mutate(c))
            else:
                c1 = choice(breeders)
                c2 = choice(breeders)
                pop.append(crossover(c1, c2))

        if no_improve_break is not None:
            last_scores.append(scores[0][0])
            if len(last_scores) == no_improve_break:
                if np.std(last_scores) == 0.0:
                    print('no improvement for %i generations' %
                          no_improve_break)
                    return scores[0][1]

    return scores[0][1]


if __name__ == "__main__":
    from time import time
    import numpy as np
    s = [1, 3, 2, 2, 3, 3, 3, 3, 2, 4, 2, 3]
    flights = load_flight_data()

    domain = [(0, 9)] * len(s)
    # bestr = genetic(domain, lambda x: schedulecost(
    #     x, flights), no_improve_break=10, maxiter=200)
    bestr = annealing_grid(domain, lambda x:schedulecost(x, flights), pop=10)
    print(bestr, schedulecost(bestr, flights), printschedule(bestr, flights))

    # N = 1000
    # with open('..//results//tests.txt', 'w') as foutput:
    #     foutput.write('test size:%i\n' % N)
    #     foutput.write('algo, time_mean, time_std, score_min, score_mean, score_max, score_std\n')
    #     opts = [randomoptimization, hillclimb, annealing, genetic]
    #     result = {}
    #     for opt in opts:
    #         print('testing %s' % opt.__name__)
    #         result[opt.__name__] = {'time': [], 'score': []}
    #         for _ in range(N):
    #             tic = time()
    #             bestr = opt(domain, lambda s: schedulecost(s, flights))
    #             toc = time() - tic

    #             result[opt.__name__]['time'].append(toc)
    #             result[opt.__name__]['score'].append(schedulecost(bestr, flights))
    #         time_mean = np.mean(result[opt.__name__]['time'])
    #         time_std = np.std(result[opt.__name__]['time'])
    #         score_mean = np.mean(result[opt.__name__]['score'])
    #         score_std = np.std(result[opt.__name__]['score'])
    #         score_min = np.min(result[opt.__name__]['score'])
    #         score_max = np.max(result[opt.__name__]['score'])
    #         line = '%s, %s, %s, %s, %s, %s, %s\n' % (opt.__name__, time_mean, time_std, score_min, score_mean, score_max, score_std)
    #         foutput.write(line )

    # printschedule(bestr, flights)
    # print(schedulecost(bestr, flights))
    # print('solve time:%s' % toc)
    # print('____________')
