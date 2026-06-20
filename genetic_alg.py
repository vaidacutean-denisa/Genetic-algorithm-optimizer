import numpy as np

class GeneticAlg:
    def __init__(self, input_file, output_file):
        self.read_file(input_file)
        self.g = open(output_file, 'w')
        self.population_info = []                                   # list of dictionaries storing info specific to each individual (such as its binary & decimal value, fitness)
        self.cumulative_prob = []

    def read_file(self, input_file):
        with open(input_file) as f:
            self.n = int(f.readline())                                            # nr of individuals
            self.x, self.y = map(float, f.readline().split())                     # domain limits
            self.a, self.b, self.c = map(float, f.readline().split())             # quadratic coefficients

            self.precision = int(f.readline())                                    # precision
            self.pc = float(f.readline())                                         # crossover probability
            self.pm = float(f.readline())                                         # mutation probability
            self.N = int(f.readline())                                            # nr of iterations

    def encode(self):
        self.l = int(np.ceil(np.log2((self.y - self.x) * 10 ** self.precision)))         # chromosome length
        self.d = (self.y - self.x) / (2 ** self.l - 1)                                   # discretization step; divide [x, y] in 2^l elements (cannot operate with inf => must divide the continuous interval into a finite nr of discrete elements)
                                                                                         # ^ 2^l possible values (search space); we only choose n values of them ->
        idx_list = np.random.randint(0, 2 ** self.l, size = self.n)                      # generate n int values in [0, 2**l - 1] (chromosomes)
        for idx in idx_list:
            chrom = bin(idx)[2:].zfill(self.l)                                  # the genotype = binary value of the index; [2:] to ignore '0b' prefix; zfill(l) to complete with 0s (for the specified length)
            val, fitness = self.evaluate_chrom(idx)

            self.population_info.append({
                "bin": chrom,
                'x': val,
                'f': fitness
            })

    def evaluate_chrom(self, chromosome):
        val = self.x + chromosome * self.d                                           # the fenotype -> linear translation (we shift the decimal value of the chromosome (without altering it) so that it's included in [x, y])
        fitness = self.f(val)                                                        # evaluates chromosome's fitness using the quadratic function f
        return val, fitness

    def f(self, val):
        return self.a * val ** 2 + self.b * val + self.c

    def compute_prob(self):
        total_fitness = sum(chrom['f'] for chrom in self.population_info)
        for chrom in self.population_info:                                              # calculates the probability of selection for each individual
            chrom['prob'] = chrom['f'] / total_fitness

        self.cumulative_prob = [0]                                                      # q(0) = 0
        prob_sum = 0
        for chrom in self.population_info:                                              # must define the 'length' of each chromosome on the [0,1] interval (needed for the method of selection by roulette wheel)
            prob_sum += chrom['prob']                                                   # ^ that means we divide [0,1] into n segments, where len(segment[i]) = chrom[i].probability; we define the length of a segment [a,b] as b-a
            self.cumulative_prob.append(prob_sum)                                       # the cumulative sum q(i) helps us find the margins of each segment

    def selection(self):
        self.compute_prob()
        elite = max(self.population_info, key = lambda chrom: chrom['f'])                   # the individual with the best fitness is preserved to the next generation (elitism)

        new_population = [elite.copy()]
        selection_values = []                                                               # stores information to be printed in the first iteration (assignment requirement)
        for _ in range(self.n - 1):                                                         # choose individuals for the remaining n-1 slots using roulette wheel selection (fitness proportionate selection)
            u = np.random.uniform(0, 1)                                                     # that means generating a random uniform value u between [0,1]
            idx = 0                                                                         # ^ followed by finding the index j for which: q(j-1) = p(1)+..+p(j-1) <= u <= q(j) = p(1)+..+p(j) (select the individual whose cumulative range includes u)

            left, right = 0, len(self.cumulative_prob) - 1
            while left <= right:                                                            # we use binary search to find the interval of [q(j), q(j + 1)); we choose the (j+1)th individual (of index j)
                mid = left + (right - left) // 2
                if self.cumulative_prob[mid] <= u:
                    idx = mid                                                               # u might be included in [q(mid), q(mid + 1)); store mid as a potential answer
                    left = mid + 1
                else:
                    right = mid - 1
            new_population.append(self.population_info[idx].copy())                         # add a copy of chrom[idx] to the new population
            selection_values.append((u, idx))

        self.population_info = new_population
        return selection_values

    def crossover(self):
        selected_chromosomes = []
        crossover_values = []                                                               # stores pairs of (index, u) that must be printed in the first iteration (assignment requirement)

        for i in range(1, self.n):                                                          # the first individual of the (new) population is the elite; it must be preserved into the next generation unchanged, therefore it's not included in crossover/mutation
            u = np.random.uniform(0, 1)                                                     # each individual receives a random uniform value between [0,1]; based on this value we decide whether an individual participates in crossover or not
            crossover_values.append((i, u))
            if u < self.pc:                                                                 # an individual is eligible for crossover if its associated value (u) is less than the probability of crossover
                selected_chromosomes.append(i)                                              # keep record of the indices of said individuals

        recombination_info = []                                                             # stores information to be printed in the first iteration
        for i in range(1, len(selected_chromosomes), 2):
            idx1 = selected_chromosomes[i - 1]                                              # create pairs of eligible individuals; one may be included at most once (if len(selected_chromosomes) % 2 != 0, an individual remains single)
            idx2 = selected_chromosomes[i]

            parent1_bin = self.population_info[idx1]['bin']
            parent2_bin = self.population_info[idx2]['bin']

            k = np.random.randint(1, self.l)                                            # single-point crossover (random point selected between 1 and chromosome length)
            child1_bin = parent1_bin[:k] + parent2_bin[k:]
            child2_bin = parent2_bin[:k] + parent1_bin[k:]

            recombination_info.append((idx1, idx2, parent1_bin, parent2_bin, k, child1_bin, child2_bin))

            self.population_info[idx1]['bin'] = child1_bin                                  # the parent individuals are replaced by their descendants
            self.population_info[idx2]['bin'] = child2_bin

            self.update_individual(idx1)                                                    # must recalculate & update the decimal values & fitness of the descendants
            self.update_individual(idx2)

        return crossover_values, recombination_info

    def update_individual(self, idx):
        dec_value = int(self.population_info[idx]['bin'], 2)
        x_val, fitness = self.evaluate_chrom(dec_value)

        self.population_info[idx]['x'] = x_val
        self.population_info[idx]['f'] = fitness

    def mutation(self):
        mutation_idx = []
        for i in range(1, self.n):                                                          # idx range starts at 1 because the first individual is the elite (explained this in the crossover method)
            modified = False
            current_chrom = list(self.population_info[i]['bin'])                            # storing the binary value as a list of bits makes it easier to perform bit flips

            for j in range(self.l):
                u = np.random.uniform(0, 1)
                if u < self.pm:
                    current_chrom[j] = '0' if current_chrom[j] == '1' else '1'
                    modified = True
            if modified:
                self.population_info[i]['bin'] = ''.join(current_chrom)
                self.update_individual(i)
                mutation_idx.append(i)

        return mutation_idx

    def write_helper(self):                                                                 # print the current population
        for idx, elem in enumerate(self.population_info):
            line = f"{idx + 1}: {elem['bin']}, x = {elem['x']}, f = {elem['f']}\n"
            self.g.write(line)

    def first_iteration(self):
        self.encode()
        self.g.write("Initial population:\n")
        self.write_helper()

        self.compute_prob()
        self.g.write("\nSelection probabilities:\n")
        for idx, chrom in enumerate(self.population_info):
            self.g.write(f"chromosome: {idx + 1}, probability: {chrom['prob']}\n")

        selection_values = self.selection()

        self.g.write("\nSelection probability intervals:\n")
        for val in self.cumulative_prob:
            self.g.write(f"{val} ")
        self.g.write("\n")

        self.g.write("\nSelection process:\n")
        for u, idx in selection_values:
            self.g.write(f"u = {u}, select chromosome: {idx + 1} \n")

        self.g.write('\n')
        self.g.write("After selection:\n")
        self.write_helper()

        self.g.write('\n')
        self.g.write(f"Crossover probability {self.pc}\n")

        crossover_values, recombination_info = self.crossover()
        for i, u in crossover_values:
            line = f"{i + 1}: {self.population_info[i]['bin']}, u = {u}"
            if u < self.pc:
                line += f" < {self.pc}, participates"
            self.g.write(line + '\n')
        self.g.write('\n')

        for idx1, idx2, p1_bin, p2_bin, k, child1_bin, child2_bin in recombination_info:
            self.g.write(f"Recombination between chromosome {idx1 + 1} and chromosome {idx2 + 1}:\n")
            self.g.write(f"{p1_bin} {p2_bin} point {k}\n")
            self.g.write(f"Result {child1_bin} {child2_bin}\n\n")

        self.g.write(f"After crossover:\n")
        self.write_helper()
        self.g.write('\n')

        mutation_idx = self.mutation()
        self.g.write(f"Mutation probability for each gene: {self.pm}\n\n")
        self.g.write("Modified chromosomes:\n")
        for idx in mutation_idx:
            self.g.write(f"{idx + 1}\n")

        self.g.write('\n')
        self.g.write("After mutation:\n")
        self.write_helper()
        self.g.write('\n')

    def main(self):
        self.first_iteration()

        self.g.write("Maximum evolution:\n")
        for i in range(1, self.N):
            self.selection()
            self.crossover()
            self.mutation()

            best_chrom = max(self.population_info, key = lambda chrom: chrom['f'])
            avg_fitness = sum(chrom['f'] for chrom in self.population_info) / self.n
            self.g.write(f"Stage {i + 1}: max: {best_chrom['f']}, average: {avg_fitness}\n")

        self.g.close()

alg = GeneticAlg("input.txt", "output.txt")
alg.main()
