# DGEANN
DGEANN (Diploid Genetics Evolving Artificial Neural Networks) is a neural network package for Python 3 built on PyCaffe. It is focused on using a diploid genetics structure to create and evolve networks, though haploidy is also supported. While it is designed to be used with the [Garden Alife simulation](https://github.com/Reedy-C/tea-garden), it is a standalone package.

DGEANN requires PyCaffe.

Networks in DGEANN consist of one pair of chromosomes that control the network layer structure and one pair of chromosomes that define the weights of the network. DGEANN can:
* turn a network with randomized weights into a genome
* turn a defined genome into a complete network
* evolve both layer structures and weight values
* perform recombination, crossing over an individual parent's pairs of chromosomes at one random point

Test coverage is currently 94%.
