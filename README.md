# Genetic Algorithm Optimizer

A Python implementation of a Genetic Algorithm (GA) designed to find the maximum value of a quadratic function within a specified domain. The implementation demonstrates binary chromosome encoding, roulette-wheel selection, single-point crossover, mutation and elitism.

## Features:
- **Binary encoding**:\
Each candidate solution is represented as a fixed-length binary chromosome. Chromosomes encode discrete indices that are mapped to real values within the search interval.

- **Continuous search space discretization**:\
  The algorithm optimizes a function over a continuous domain by discretizing the interval according to a user-defined precision and encoding the resulting values as binary strings.

- **Fitness evaluation**:\
  Each chromosome is decoded into its corresponding real value and evaluated using the quadratic fitness function
  $$f(x) = ax^2 + bx + c$$

- **Roulette-wheel selection**:\
  Individuals are selected for reproduction using fitness-proportionate selection based on their cumulative probabilities. This ensures that the likelihood of choosing a candidate is proportional to its fitness, giving better solutions a higher chance of being chosen while preserving population diversity.

- **Elitism**:\
  The best individual of each generation is automatically preserved to the next generation to ensure convergence.

- **Single-point crossover**:\
  Selected parent chromosomes exchange genetic material at a randomly chosen crossover point, producing new offspring that combine traits from both parents.

- **Bit-flip mutation**:\
  Each gene (bit) is subject to a mutation probability specified in the input file. Mutation introduces randomness into the population and helps prevent premature convergence to a local optimum.

- **Efficient selection using binary search**:\
  The roulette-wheel selection process implements binary search to efficiently navigate the cumulative probability distribution, reducing the computational cost of locating the selected individual to $O(logn)$.

- **Evolution monitoring**:\
  The algorithm records the maximum fitness and average population fitness at every generation, making it possible to observe the evolutionary progress over time.

## Detailed first-generation report 
For educational and debugging purposes, the program provides a complete breakdown of the first generation, including:
- Initial population
- Fitness values
- Selection probabilities
- Cumulative probability intervals
- Selection decisions
- Crossover participants
- Recombination results
- Mutated chromosomes

## Configurable parameters
All parameters can be adjusted through the input file:
- population size
- search interval
- quadratic function coefficients
- precision
- crossover probability
- mutation probability
- number of generations

*Note: throughout the implementation, the terms "chromosome" and "individual" are used interchangeably. When referencing an entity within `population_info`, both terms refer to the same individual.
