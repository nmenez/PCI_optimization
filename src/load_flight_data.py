from time import strptime

people = [('Seymour', 'BOS'),
          ('Franny', 'DAL'),
          ('Zooey', 'CAK'),
          ('Walt', 'MIA'),
          ('Buddy', 'ORD'),
          ('Les', 'OMA')]
destination = 'LGA'


def load_flight_data():
    flights = {}
    with open('..//data//schedule.txt') as f:
        for line in f:
            origin, dest, depart, arrive, price = line.strip().split(',')
            flights.setdefault((origin, dest), [])
            flights[(origin, dest)].append((depart, arrive, int(price)))

    return flights


def printschedule(r, flights):
    for d, (name, origin) in enumerate(people):

        out = flights[(origin, destination)][r[2 * d]]
        ret = flights[(origin, destination)][r[2 * d + 1]]

        str_ = '%10s%10s %5s-%5s $%3s %5s-%5s $%3s' %\
               (name, origin, out[0], out[1], out[2],
                ret[0], ret[1], ret[2])
        print(str_)


if __name__ == "__main__":
    flights = load_flight_data()
    s = [1, 3, 2, 2, 3, 3, 3, 3, 2, 4, 2, 3]

    printschedule(s, flights)
