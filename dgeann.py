import copy
import math
import os
import random
from textwrap import dedent

import caffe
import numpy


#default solver
solv = '''\
        net: "{0}"
        type: "AdaDelta"
        momentum: 0.95
        base_lr: 0.2
        lr_policy: "step"
        gamma: 0.1
        stepsize: 1000000
        max_iter: 2000000
        display: 10000
        '''
#sigma for mutation rate mutations
sigma = 0.001

#default mutation rate used for turning random weights into weight genes
def_mut_rate = 0.01

#constrains crossover in to try and not break network substructures
#not sure if necessary?
#TODO test further
constrain_crossover = True

#dict for layer types, used to generate caffe network def files
layer_dict = {"input":'''\
                        input: "{0.ident}"
                        input_shape: {{
                          dim: 1
                          dim: {0.nodes}
                        }}
                        ''',
              "concat_0":'''\
                            layer {{
                              name: "{0}"
                              type: "Concat"
                              ''',
              "concat_1": '''\
                              top: "{0}"
                              concat_param {{
                                axis: 1
                              }}
                            }}
                            ''',
                "IP":'''\
                        layer {{
                          name: "{0.ident}"
                          type: "InnerProduct"
                          param {{ lr_mult: 1 decay_mult: 1}}
                          param {{ lr_mult: 2 decay_mult: 0}}
                          inner_product_param {{
                            num_output: {0.nodes}
                            weight_filler {{
                              type: "xavier"
                            }}
                            bias_filler {{
                              type: "constant"
                              value: 0
                            }}
                          }}
                          bottom: "{0.inputs[0]}"
                          top: "{0.ident}"
                        }}
                        ''',
                "loss": '''\
                    layer {{
                      name: "{0.ident}"
                      type: "EuclideanLoss"
                      bottom: "{0.inputs[0]}"
                      bottom: "{0.inputs[1]}"
                      top: "{0.ident}"
                    }}
                    '''}

#this is used by determine_mutation for layer genes only
#1 is probability of a dominance mutation
#2 is probability of a mutation rate mutation
#3 is probability of a number of nodes mutation
#4 is probability of a duplication mutation
#5 is probability of an add input mutation (currently not implemented)
#this can be changed, but numbers should add up to 1
#(or mutations will not happen correctly)
#current probabilities were basically made up on the spot
layer_mut_probs = (0.333, 0.234, 0.167, 0.133, 0.133)

#same as above, but for weight gene mutations
#1 is probability of a weight mutation
#2 is probability of a dominance mutation
#3 is probability of a mutation rate mutation
weight_mut_probs = (0.833, 0.117, 0.05)

#toggles recording of mutations in child from parents
record_muts = True


class Genome(object):
    """Genome defining a neural network.

    A genome is a list of lists of genes referred to as chromosomes.
    These genomes are diploid, so there are two copies of each chromosome,
    which may have different genes on them.
    Chromosome pair 1: layer genes; pair 2: weight genes.
    Outs: optional list of output/top-level layers.
    Mut_record: record of mutations from parents, if toggled.
    """

    def __init__(self, layerchr_a, layerchr_b, weightchr_a, weightchr_b,
                 outs = None):
        self.layerchr_a = layerchr_a
        self.layerchr_b = layerchr_b
        self.weightchr_a = weightchr_a
        self.weightchr_b = weightchr_b
        self.outs = outs
        self.mut_record = []

    def recombine(self, other_genome):
        """Return a new child genome from two parent genomes.
        """
        #first we need to do crossover on each genome
        parent_one = self.crossover()
        parent_two = other_genome.crossover()
        #then randomly pick chromosome for layers and weights from each
        layers = random.randint(0, 1)
        if layers == 0:
            layer_one = parent_one.layerchr_a
            layer_two = parent_two.layerchr_b
        else:
            layer_one = parent_one.layerchr_b
            layer_two = parent_two.layerchr_a
        weights = random.randint(0, 1)
        if weights == 0:
            weight_one = parent_one.weightchr_a
            weight_two = parent_two.weightchr_b
        else:
            weight_one = parent_one.weightchr_b
            weight_two = parent_two.weightchr_a
        layer_one = copy.deepcopy(layer_one)
        layer_two = copy.deepcopy(layer_two)
        weight_one = copy.deepcopy(weight_one)
        weight_two = copy.deepcopy(weight_two)
        for gen in weight_one:
            gen.alt_in = gen.in_node
        for gen in weight_two:
            gen.alt_in = gen.in_node
        child = Genome(layer_one, layer_two, weight_one, weight_two)
        #now just do mutations
        child.mutate()
        return child

    def crossover(self):
        """Return a new genome with both pairs of chromosomes crossed over.
        """
        if constrain_crossover:
            #n = last possible point of crossover for layer chros
            #m = last possible point of crossover for weight chros
            n, m = self.last_shared()
        else:
            n = min(len(self.layerchr_a)-1, len(self.layerchr_b)-1)
            m = min(len(self.weightchr_a)-1, len(self.weightchr_b)-1)
        s_diffs = self.find_size_diffs()
        lay_cross = random.randint(1, n)
        layer_a = []
        layer_b = []
        for x in range(0, lay_cross):
            layer_a.append(self.layerchr_a[x])
        for x in range(lay_cross, len(self.layerchr_b)):
            layer_a.append(self.layerchr_b[x])
        for x in range(0, lay_cross):
            layer_b.append(self.layerchr_b[x])
        for x in range(lay_cross, len(self.layerchr_a)):
            layer_b.append(self.layerchr_a[x])
        weight_a, weight_b = self.cross_weights(s_diffs, m)
        result = Genome(layer_a, layer_b, weight_a, weight_b)
        return result

    #helper function for crossover
    def last_shared(self):
        """Return the last possible crossover points for layer and weight
        chromosomes.

        Tries to minimize the wrecking of possible evolved layer strucutres,
        hence why it's not just randint(0, len(chromosome)-1).
        """
        n = 0
        m = min(len(self.weightchr_a), len(self.weightchr_b))
        last_layer = ""
        while n < (len(self.layerchr_a)-1) and n < (len(self.layerchr_b)-1):
            n += 1
            if self.layerchr_a[n].ident == self.layerchr_b[n].ident: 
                if (self.layerchr_a[n].layer_type == "IP" or
                        self.layerchr_a[n].layer_type == "input"):
                    last_layer = self.layerchr_a[n].ident
            else:
                #if we get one mismatch, check the next one
                #assuming there's still genes left
                #and continue if THOSE match
                #otherwise, it's time to break
                #this allows for one mutation in a streak of shared ancestry
                #but not for having a whole series of mismatches
                if (n+1) < len(self.layerchr_a) and \
                   (n+1) < len(self.layerchr_b):
                    if self.layerchr_a[n+1].ident == self.layerchr_b[n+1].ident:
                        n += 1
                        if (self.layerchr_a[n+1].layer_type == "IP" or
                                self.layerchr_a[n+1].layer_type == "input"):
                            last_layer = self.layerchr_a[n].ident
                    else:
                        n -= 1
                        break
                else:
                    n -= 1
                    break
        #as for the weight genes, as much as possible we don't want to have
        #a layer that uses them (data/IP layers) but no weights defined
        #so, we want the last crossover point be at the point where these are
        while m > 0:
            m -= 1
            if (self.weightchr_a[m].in_layer == last_layer or
                    self.weightchr_a[m].out_layer == last_layer):
                break
            elif (self.weightchr_a[m].in_layer == last_layer or
                      self.weightchr_b[m].out_layer == last_layer):
                break
        return n, m

    #helper function for crossover
    def find_size_diffs(self):
        """Return any shared layers between chromosomes that have different
        defined sizes. Used to prevent weight genes getting lost.
        """
        n = 0
        s_diffs = {}
        ips = {}
        for gen in self.layerchr_a:
            if gen.layer_type == "IP":
                ips[gen.ident] = gen.nodes
        for gen in self.layerchr_b:
            if gen.layer_type == "IP":
                if gen.ident in ips.keys():
                    if gen.nodes != ips[gen.ident]:
                        s_diffs[gen.ident] = max(gen.nodes, ips[gen.ident])
        return s_diffs

    #helper function for crossover
    def cross_weights(self, s_diffs, m):
        """Return two crossed-over weight chromosomes.

        s_diffs: dictionary of any shared layers with different sizes
        m: crossover point
        """
        if m > 0:
            weight_cross = random.randint(1, m)
        else:
            weight_cross = 0
        #if there is no difference in layer sizes, easy way
        if s_diffs == {}:
            weight_a, weight_b = self.cross_weights_simple(weight_cross)
        else:
            weight_a = self.cross_weights_comp(weight_cross, self.weightchr_a,
                                               self.weightchr_b, s_diffs)
            weight_b = self.cross_weights_comp(weight_cross, self.weightchr_b,
                                               self.weightchr_a, s_diffs)
        return weight_a, weight_b

    #helper function for cross_weights
    def cross_weights_simple(self, weight_cross):
        """Return new crossed-over weight chromomes when doing so is simple.

        weight_cross: pre-determined crossover point
        """
        weight_a = []
        weight_b = []
        for x in range(0, weight_cross):
            self.weightchr_a[x].alt_in = self.weightchr_a[x].in_node
            weight_a.append(self.weightchr_a[x])
        for x in range(weight_cross, len(self.weightchr_b)):
            self.weightchr_b[x].alt_in = self.weightchr_b[x].in_node
            weight_a.append(self.weightchr_b[x])
        for x in range(0, weight_cross):
            self.weightchr_b[x].alt_in = self.weightchr_b[x].in_node
            weight_b.append(self.weightchr_b[x])
        for x in range(weight_cross, len(self.weightchr_a)):
            self.weightchr_a[x].alt_in = self.weightchr_a[x].in_node
            weight_b.append(self.weightchr_a[x])
        return weight_a, weight_b

    #helper function for cross_weights:
    def cross_weights_comp(self, weight_cross, chr_a, chr_b, s_diffs):
        """Return new crossed-over weight chromomes when a shared layer has
        different lengths on either layer chromosome.

        weight_cross: pre-determined crossover point
        """
        weights = self.cross_weights_one(0, weight_cross, chr_a, chr_b, s_diffs,
                                         [])
        weights = self.cross_weights_one(weight_cross, len(chr_b), chr_b,
                                         chr_a, s_diffs, weights)
        return weights
    
    #helper function for cross_weights_comp
    def cross_weights_one(self, start, stop, chr_a, chr_b, s_diffs, weights):
        """Return a single crossed_over weight chromosome where a shared layer
        has different lengths on either layer chromosome.

        start: starting point for crossover
        stop: ending point for crossover
        """
        cur_out = None
        cur_in = None
        out_targ = None
        find_chro_diff = False
        if start != 0:
            alt_start, weights = self.cross_weights_altstart(start, chr_a,
                                                             chr_b, weights)
            if alt_start != None:
                start = alt_start
            cur_in = weights[-1].in_node
            cur_out = weights[-1].out_node
            if chr_a[start].out_layer in s_diffs.keys():
                find_chro_diff = True
                out_targ = s_diffs[chr_a[start].out_layer] - 1
        in_targ = None
        #decided not to have orphaned weights w/ no genes to cover them
        for x in range(start, stop):
            #covers cases where we switch at the very end
            #in which case, current node *should* already be in weights
            if stop - start == 1:
                break
            cur = chr_a[x]
            if cur.in_node == 0 and cur.out_node == 0:
                #if we didn't finish outs of previous layer
                if cur_out != None:
                    last = weights[-1]
                    if last.out_layer in s_diffs:
                        if last.out_node != s_diffs[last.out_layer] - 1:
                            #TODO extract function
                            need_search = True
                            if start != 0:
                                #check back if we switched
                                if chr_a[x-1].out_node > weights[-1].out_node:
                                    search = x
                                    while chr_a[search].out_node != out_targ:
                                        search -= 1
                                    while search != x:
                                        weights.append(chr_a[search])
                                        search += 1
                                    need_search == False
                            if need_search == True:
                                for y in range(len(chr_b)):
                                    oth = chr_b[y]
                                    if oth.in_layer == last.in_layer and \
                                       oth.out_layer == last.out_layer:
                                        if oth.in_node == last.in_node and \
                                           oth.out_node == last.out_node + 1:
                                            while (oth.out_layer ==
                                                   last.out_layer and \
                                                   oth.out_node <=
                                                   (s_diffs[oth.out_layer] - 1)):
                                                oth.alt_in = oth.in_node
                                                weights.append(oth)
                                                y += 1
                                                if y < len(chr_b):
                                                    oth = chr_b[y]
                                                else:
                                                    break
                                            break
                    cur_out = None
                #if we didn't finish ins of previous layer
                if cur_in != None:
                    last = chr_a[x-1]
                    for y in range(len(chr_b)):
                        oth = chr_b[y]
                        if oth.in_layer == last.in_layer and \
                           oth.out_layer == last.out_layer:
                            if oth.in_node == last.in_node + 1:
                                while oth.in_node <= (s_diffs[
                                    oth.in_layer] - 1) and \
                                    oth.in_layer == last.in_layer:
                                        oth.alt_in = oth.in_node
                                        weights.append(oth)
                                        y += 1
                                        if y < len(chr_b):
                                            oth = chr_b[y]
                                        else:
                                            break
                                break
                    #check if we need to make any extra genes
                    cur_in = None
                    if oth.out_layer in s_diffs.keys():
                        if oth.out_node == (s_diffs[oth.out_layer] - 1):
                            cur_out = None
                        else:
                            pass
                    else:
                        cur_out = None
            if cur.out_layer in s_diffs.keys():
                if cur.in_node == 0 and cur.out_node == 0:
                    chro_diff = None
                    nodes_diff = None
                    out_targ = s_diffs[cur.out_layer] - 1
                    weights.append(cur)
                    cur_out = cur.out_node
                #first, if we have moved on to a 1, 0
                elif cur.in_node == 1 and cur.out_node == 0:
                    #check if we have met out_targ
                    if chr_a[x-1].out_node == out_targ:
                        weights.append(cur)
                        cur_in = cur.in_node
                        cur_out = None
                    else:
                        nodes_diff = out_targ - chr_a[x-1].out_node
                        #if not, then find 0, [necessary] nodes in b and add
                        for y in range(len(chr_b)):
                            b_cur = chr_b[y]
                            if b_cur.in_layer == cur.in_layer and \
                               b_cur.out_layer == cur.out_layer and\
                               b_cur.in_node == 0 and\
                               b_cur.out_node == chr_a[x-1].out_node + 1:
                                chro_diff = y - x
                                for i in range(nodes_diff):
                                    if weights[-1].out_node != out_targ:
                                                weights.append(chr_b[y + i])
                                    if y + i + 1 > len(chr_b):
                                        break
                                break
                        if start != 0:
                            find_chro_diff = False
                        weights.append(cur)
                        cur_in = cur.in_node
                        cur_out = cur.out_node
                else:    
                    skip = False
                    if start != 0:
                        #do not duplicate a weight we've already done
                        if cur.in_node == weights[-1].in_node and \
                           cur.out_node == weights[-1].out_node:
                            if cur.in_layer == weights[-1].in_layer and \
                               cur.out_layer == weights[-1].out_layer:
                                skip = True
                    if skip == False:
                        if cur.out_node == 0 and cur.in_node != 0:
                            #if we haven't already established chro_diff
                            #due to starting further into the genes
                            if find_chro_diff == True:
                                chro_diff = None
                                nodes_diff = out_targ - \
                                             chr_a[x-1].out_node
                                for y in range(len(chr_b)):
                                    b_cur = chr_b[y]
                                    if b_cur.in_layer == cur.in_layer and \
                                       b_cur.out_layer == cur.out_layer and \
                                       b_cur.in_node == cur.in_node - 1 and \
                                       b_cur.out_node == chr_a[x-1].out_node + 1:
                                        chro_diff = y - x - (chr_a[x].in_node - 1)
                                        for i in range(nodes_diff):
                                            if weights[-1].out_node != out_targ:
                                                weights.append(chr_b[y + i])
                                            if y + i + 1 > len(chr_b):
                                                break
                                        break
                                if cur.in_node == out_targ:
                                    cur_in = None
                                else:
                                    cur_in = cur.in_node
                                cur_out = cur.out_node
                                find_chro_diff = False
                                #...and check to make sure we don't need to
                                #finish from *a*
                                if chro_diff == None:
                                    if weights[-1].out_node < out_targ:
                                        search = x
                                        while chr_a[search].out_node != out_targ:
                                            search -= 1
                                        while search != x:
                                            weights.append(chr_a[search])
                                            search += 1
                                weights.append(cur)
                            else:
                                if chro_diff != None:
                                    b_start = (x + chro_diff +
                                             (nodes_diff * cur.in_node) - 1)
                                    for i in range(nodes_diff):
                                        weights.append(chr_b[b_start + i])
                                        if b_start + i + 1 > len(chr_b):
                                            break
                                    weights.append(cur)
                                else:
                                    weights.append(cur)
                                    if cur.in_node == out_targ:
                                        cur_in = None
                                    else:
                                        cur_in = cur.in_node
                                    cur_out = cur.out_node
                        else:
                            weights.append(cur)
                            cur_out = cur.out_node
            else:
                weights.append(cur)
        if start != 0:
            #check again if we need to finish on b?
            if weights[-1].in_layer in s_diffs.keys():
                n_node = None
                for i in range(len(chr_b)):
                    n = chr_b[i]
                    if n.in_node > weights[-1].in_node and n.out_node == 0 \
                       and n.in_layer == weights[-1].in_layer and \
                       n.out_layer == weights[-1].out_layer:
                        n_node = n
                        break
                if n_node != None:
                    while n_node.in_node < s_diffs[weights[-1].in_layer]:
                        weights.append(n)
                        i += 1
                        if i < len(chr_b):
                            n = chr_b[i]
                        else:
                            break
                #lastly, check if we need to make any extra out weights
                if weights[-1].out_layer in s_diffs.keys():
                    lim = s_diffs[weights[-1].out_layer] - 1
                    if weights[-1].out_node < lim:
                        for i in range(weights[-1].out_node + 1, lim + 1):
                            #TODO
                            #real Xavier initialization later
                            w = random.gauss(0, 0.1)
                            weights.append(WeightGene(random.randint(1, 5),
                                                      True, False, def_mut_rate,
                                                      gene_ident(), w,
                                                      weights[-1].in_node,
                                                      i, weights[-1].in_layer,
                                                      weights[-1].out_layer))
        return weights

    #helper function for cross_weights_one
    def cross_weights_altstart(self, start, chr_a, chr_b, weights):
        """Return a possible alternate starting point and the weights genome,
        when starting the second half of crossover for unequal chromosomes.
        """
        #see if we need to finish something from a (now b) that is not in b
        #or adjust start
        new_start = start
        if chr_a[start].in_layer != chr_b[start].in_layer or \
           chr_a[start].out_layer != chr_b[start].out_layer:
            #find if connection is in A (old B)
            if chr_a[start].in_layer == chr_b[start].out_layer:
                #then we need to move back
                while chr_a[new_start].in_layer != chr_b[start].in_layer or \
                      chr_a[new_start].out_layer != chr_b[start].out_layer:
                        if new_start - 1 > 0:
                            new_start -= 1
                        else:
                            #probably not on this chro
                            new_start = start
                            break
            elif chr_a[start].out_layer == chr_b[start].in_layer:
                #then we need to move forward
                while chr_a[new_start].in_layer != chr_b[start].in_layer or \
                      chr_a[new_start].out_layer != chr_b[start].out_layer:
                    if new_start + 1 < len(chr_a) - 1:
                        new_start += 1
                    else:
                        #probably not on this chro
                        new_start = start
                        break
            else:
                #start going forward, loop if necessary
                if new_start + 1 < len(chr_a) - 1:
                    new_start += 1
                    while chr_a[new_start].in_layer != chr_b[start].in_layer or \
                      chr_a[new_start].out_layer != chr_b[start].out_layer:
                        if new_start + 1 < len(chr_a) - 1:
                            new_start += 1
                        elif new_start == start:
                            #not found
                            break
                        else:
                            new_start = 0
        if chr_a[new_start].in_node > chr_b[start].in_node:
            while chr_a[new_start].in_node != chr_b[start].in_node:
                new_start -= 1
            while chr_a[new_start].out_node > chr_b[start].out_node:
                new_start -= 1
        elif chr_a[new_start].in_node < chr_b[start].in_node:
            while chr_a[new_start].in_node != chr_b[start].in_node:
                if new_start + 1 < len(chr_a) - 1:
                    new_start += 1
                else:
                    #then we might need to finish up from chr_b
                    break
            x_start = new_start
            while chr_a[x_start].out_node < weights[-1].out_node:
                if chr_a[x_start].in_node != weights[-1].in_node:
                    x_start = new_start
                    break
                else:
                    if x_start + 1 < len(chr_a) - 1:
                        x_start += 1
                    else:
                        break
            if  x_start + 1 < len(chr_a) - 1 and \
                chr_a[x_start + 1].out_node == weights[-1].out_node + 1:
                x_start += 1
            new_start = x_start
        elif chr_a[new_start].out_node != chr_b[start].out_node:
            if chr_a[new_start].out_node < chr_b[start].out_node:
                while chr_a[new_start].out_node != chr_b[start].out_node:
                    if new_start + 1 < len(chr_a) - 1:
                        new_start += 1
                    else:
                        break
            else:
                while chr_a[new_start].out_node != chr_b[start].out_node:
                    new_start -= 1
        if start == new_start:
            return None, weights
        else:
            return new_start, weights

    #TODO is it possible to simplify and get rid of active_list
    #   given that I now know that list(t._layer/blob_names) exists?
    def build(self, delete=True):
        """Return the solver for the PyCaffe network from the Genome.

        Delete: if true, deletes the generated solver files.
        """
        #first, generate a new ID for the network
        self.ident = network_ident()
        if not os.path.exists('Gen files'): # pragma: no cover
            os.makedirs('Gen files')
        ident_file = os.path.join('Gen files', self.ident + '.gen')
        #then build network structure
        active_list = {}
        concat_dict = {}
        active_list, concat_dict, sub_dict = self.build_layers(active_list,
                                                               ident_file,
                                                               concat_dict)
        result = dedent(solv.format(ident_file))
        f = open("temp_solver.txt", "w")
        f.write(result)
        f.close()
        #TODO make replaceable with other solvers
        solver = caffe.AdaDeltaSolver('temp_solver.txt')
        os.remove('temp_solver.txt')
        if delete == True:
            os.remove(ident_file)
        #deal with concats and weights
        self.concat_adjust(concat_dict)
        #now change the weights to those specified in genetics
        if len(self.weightchr_a) > 0:
            self.build_weights(active_list, solver.net, sub_dict)
        else:
            self.rand_weight_genes(solver.net, concat_dict)
        return solver

    #helper function for build
    def build_layers(self, active_list, ident_file, concat_dict):
        """Create the file with the layer structure of the network
        defined by the genome, and return active_list, concat_dict, and sub_dict.
        """
        if len(self.layerchr_b) != 0:
            self.layers_equalize()
        #(if genome is actually haploid)
        else:
            for i in range(len(self.layerchr_a)):
                self.layerchr_b.append(LayerGene(0, False, False, 0, "null",
                                                  [], None, None))
                i += 1
        sub_dict, active_list, layout = self.structure_network(active_list)
        #read out combined genome
        for gene in layout:
            print_out = gene.read_out(concat_dict, active_list)
            #print out to file
            f = open(ident_file, "a")
            f.write(print_out)
            f.close()
        return active_list, concat_dict, sub_dict

    #helper function for build_layers
    def layers_equalize(self):
        """Force the two layer chromosomes to be an equal length by
        inserting null layers into the shorter one.
        """
        if len(self.layerchr_a) != len(self.layerchr_b):
            n = 0
            m = 0
            #run through backwards
            self.layerchr_a.reverse()
            self.layerchr_b.reverse()
            while len(self.layerchr_a) != len(self.layerchr_b):
                #see if a == b
                if self.layerchr_a[n].ident != self.layerchr_b[m].ident:
                #if not, then insert a new null layer
                #under the shorter one
                    null = LayerGene(3, False, False, 0, "null", [],
                                      None, None)
                #compare same layer in shorter with next in longer
                    if len(self.layerchr_a) < len(self.layerchr_b):
                        self.layerchr_a.insert(self.layerchr_a.index(
                                                self.layerchr_a[n]),
                                               null)
                    else:
                        self.layerchr_b.insert(self.layerchr_b.index(
                                                self.layerchr_b[m]),
                                               null)
                n += 1
                m += 1
            self.layerchr_a.reverse()
            self.layerchr_b.reverse()

    #helper function for build_layers
    def structure_network(self, active_list):
        """Return a list of genes that are ready to be turned into a
        Caffe network file, active_list ({layer: # nodes}), and substitution
        dictionary.
        """
        #choose one layer chr to use as layout structure pattern
        layout = copy.copy(random.choice([self.layerchr_a,
                                          self.layerchr_b]))
        orphan_list = []
        sub_dict = {}
        del_list = []
        for i in range(len(self.layerchr_a)):
            #for pair in chrs: read
            read_gene = self.layerchr_a[i].read(active_list,
                                                self.layerchr_b[i], sub_dict,
                                                del_list)
            if read_gene is not None:
            #if gene already there in layout: keep
                if read_gene == layout[i]:
                    if read_gene.ident != 'null':
                        layout[i] = copy.deepcopy(read_gene)
                        ins = []
                        for lay in layout[i].inputs:
                            if lay in sub_dict:
                                ins.append(sub_dict[lay])
                            else:
                                if lay in active_list and lay not in del_list:
                                    ins.append(lay)
                        layout[i].inputs[:] = [a for a in ins]
                        active_list[read_gene.ident] = read_gene.nodes
                        orphan_list.append(read_gene.ident)
                        for j in read_gene.inputs:
                            if j in orphan_list:
                                orphan_list.remove(j)
                #else if other gene, if not null
                else:
                    if read_gene.ident != 'null':
                    #keep inputs the same if possible, but sub name in outputs
                        new_layer = copy.deepcopy(read_gene)
                        if layout[i].ident != 'null':
                            new_layer.inputs = layout[i].inputs
                        ins = []
                        for lay in new_layer.inputs:
                            if lay in sub_dict:
                                ins.append(sub_dict[lay])
                            else:
                                if lay in active_list and lay not in del_list:
                                    ins.append(lay)
                        new_layer.inputs[:] = [a for a in ins]
                        if layout[i].ident != new_layer.ident:
                            sub_dict = {layout[i].ident: new_layer.ident}
                        layout[i] = new_layer
                        orphan_list.append(layout[i].ident)
                        active_list[layout[i].ident] = layout[i].nodes
                        for j in layout[i].inputs:
                            if j in orphan_list:
                                orphan_list.remove(j)
                    else:
                        del_list.append(layout[i].ident)
                        layout[i] = read_gene
            #adjust weights to use A's weights with B's name?
            #if null OR None (?), delete names in output
            #TODO skipping with None for now
            else:
                if layout[i].ident != 'null':
                    layout[i] = LayerGene(3, False, False, 0, "null", [],
                                            None, None)
        #clear out orphans
        if self.outs != None:
            for i in range(len(layout)):
                if (layout[i].ident in orphan_list
                    and layout[i].ident not in self.outs):
                    layout[i] = LayerGene(3, False, False, 0, "null", [],
                                                None, None)
        #now delete all null layers in chr a, chr b, and layout
        layout[:] = [x for x in layout if x.ident != "null"]
        self.layerchr_a[:] = [x for x in self.layerchr_a if x.ident != "null"]
        self.layerchr_b[:] = [x for x in self.layerchr_b if x.ident != "null"]
        return sub_dict, active_list, layout

    def concat_adjust(self, concat_dict):
        """Adjust offsets for weight gene input nodes when concat layers
        exist, so that the correct weights are adjusted in the final network.
        """
        #probably not the fastest way to do this
        for ch in [self.weightchr_a, self.weightchr_b]:
            for weight in ch:
                for key in concat_dict:
                    if (weight.in_layer in concat_dict[key][0] and
                        weight.out_layer in concat_dict[key][2]):
                        #don't adjust the first one
                        if (concat_dict[key][0].index(weight.in_layer)) != 0:
                            n = concat_dict[key][1][0]
                            for lay in concat_dict[key][1][
                                1:(concat_dict[key][0].index(weight.in_layer))]:
                                n += lay
                            weight.alt_in = n + weight.in_node

    def build_weights(self, active_list, net, sub_dict):
        """Change the weights in the created network to those defined
        by the weight genes.
        """
        weightchr_a = self.weightchr_a
        weightchr_b = self.weightchr_b
        if len(weightchr_a) < len(weightchr_b):
            longer = weightchr_b
        elif len(weightchr_a) > len(weightchr_b):
            longer = weightchr_a
        else:
            longer = None
        n = 0
        m = 0
        a_lim = len(weightchr_a) - 1
        b_lim = len(weightchr_b) - 1
        while n <= a_lim and m <= b_lim:
            #first see if the two strands are in sync
            a = weightchr_a[n]
            b = weightchr_b[m]
            if a.in_node == b.in_node:
                if a.out_node == b.out_node:
                    values = a.read(active_list, sub_dict, b)
                    if values is not None:
                        Genome.adjust_weight(net, values)
                    n += 1
                    m += 1
                    if n > a_lim and m <= b_lim:
                        longer = weightchr_b
                    elif m > b_lim and n <= a_lim:
                        longer = weightchr_a
                else:
                    #inputs are at the same node but outputs are not
                    #if one is at 0/0 and the other is not:
                    if a.in_node == 0 and a.out_node == 0:
                        m, b, longer = self.catch_out("b", b, b_lim, m,
                                                      active_list, net,
                                                      sub_dict, longer)
                    elif b.in_node == 0 and b.out_node == 0:
                        n, a, longer = self.catch_out("a", a, a_lim, n,
                                                      active_list, net,
                                                      sub_dict, longer)
                    else:
                        #can we get here? in nodes equal, out nodes not
                        #but neither is at 0/0?
                        print("somehow, got here", self.ident)
                        while a.out_node < b.out_node:
                            n, a = self.read_through("a", n, active_list, net,
                                                     sub_dict)
                            if n > a_lim:
                                longer = self.weightchr_b
                                break
                            if a.in_node == 0 and a.out_node == 0:
                                break
                        while b.out_node < a.out_node:
                            m, b = self.read_through("b", m, active_list, net,
                                                     sub_dict)
                            if m > b_lim:
                                longer = self.weightchr_a
                                break
                            if b.in_node == 0 and b.out_node == 0:
                                break
            else:
                #if one is at 0 input/0 output, read the other 'till it catches up
                if a.in_node == 0 and a.out_node == 0:
                    while b.in_node != 0:
                        m, b = self.read_through("b", m, active_list, net,
                                                 sub_dict)
                        if m > b_lim:
                            break
                    if m <= b_lim:
                        while b.out_node != 0:
                            m, b = self.read_through("b", m, active_list, net,
                                                     sub_dict)
                            if m > b_lim:
                                break
                elif b.in_node == 0 and b.out_node == 0:
                    while a.in_node != 0:
                        n, a = self.read_through("a", n, active_list, net,
                                                 sub_dict)
                        if n > a_lim:
                            break
                    if n <= a_lim:
                        while a.out_node != 0:
                            n, a = self.read_through("a", n, active_list, net,
                                                     sub_dict)
                            if n > a_lim:
                                break
                #if one is at a higher input #, read the other 'till it catches up
                #or until the other hits 0/0 i/o
                else:
                    while a.in_node < b.in_node:
                        n, a = self.read_through("a", n, active_list, net,
                                                 sub_dict)
                        if n > a_lim:
                            longer = self.weightchr_b
                            break
                        if a.in_node == 0 and a.out_node == 0:
                            break
                    while b.in_node < a.in_node:
                        m, b = self.read_through("b", m, active_list, net,
                                                 sub_dict)
                        if m > b_lim:
                            longer = self.weightchr_a
                            break
                        if b.in_node == 0 and b.out_node == 0:
                            break
        if longer is not None:
            #figure out the starting value (since we don't want to bother
            #reading genes we already read)
            if n <= a_lim or m <= b_lim:
                if n > a_lim:
                    x = m
                else:
                    x = n
                while x <= (len(longer) - 1):
                    values = longer[x].read(active_list, sub_dict)
                    if values is not None:
                        Genome.adjust_weight(net, values)
                    x += 1

    #helper function for build_weights
    @staticmethod
    def adjust_weight(net, values):
        """Change an individual weight in the network to that specified
        by a particular pair of weight genes.
        """
        #values is a list formatted as:
        #input (str), in node, output (str), out node, weight
        #with perhaps another set for a second weight adjustment
        output = values[2]
        out_node = values[3]
        in_node = values[1]
        weight = values[4]
        net.params[output][0].data[out_node][in_node] = weight
        if len(values) > 5:
            output = values[7]
            out_node = values[8]
            in_node = values[6]
            weight = values[9]
            net.params[output][0].data[out_node][in_node] = weight
            
    #helper function for build_weights
    def read_through(self, chro, n, active_list, net, sub_dict):
        """Adjusts weight genes while reading through one chromosome at a time,
        and returns position of last read gene and the gene itself.
        """
        if chro == "a":
            chro = self.weightchr_a
        elif chro == "b":
            chro = self.weightchr_b
        a = chro[n]
        values = a.read(active_list, sub_dict)
        if values is not None:
            Genome.adjust_weight(net, values)
        n += 1
        if n <= (len(chro) - 1):
            a = chro[n]
        return n, a

    #helper function for build_weights
    def catch_out(self, chrom, gene, lim, pos, active_list, net, sub_dict,
                  longer):
        """Return current position, gene, and longer chromosome after catching
        up when reading through genome, when inputs are equal, outputs aren't,
        and other current gene is at 0-in 0-out.

        chrom: "a" or "b"
        gene: gene currently being read
        lim: length limit of chromosome
        pos: current position being read
        """
        while gene.out_node != 0:
                pos, gene = self.read_through(chrom, pos, active_list, net,
                                              sub_dict)
                if pos > lim:
                    if chrom == "a":
                        longer = self.weightchr_a
                    else:
                        longer = self.weightchr_b
                    break
        return pos, gene, longer

    #helper function for build
    def rand_weight_genes(self, net, concat_dict):
        """Create a network with random weights and create weight genes for
        both chromosomes based on those weights.

        Both are given the same weights, but with random dominance.
        """
        for key in net.params:
            d = net.params[key][0].data
            if type(d[0]) == numpy.ndarray:
                #have to find in_layer, undo concats
                in_layer = list(net._blob_names)[
                    list(net._bottom_ids(list(net._layer_names).index(key)))[0]]
                #now see if this is in concat_dict?
                conc = False
                for con in concat_dict:
                    #if in_layer in concat_dict[con][0]:
                    if in_layer in concat_dict.keys():
                        #then input is from a concat layer
                        conc = True
                        break
                if conc == True:
                    self.concat_rweights(net, in_layer, d, key, concat_dict)
                else:
                    self.create_rweights(in_layer, d, key, net)

    #helper function for rand_weight_genes
    #d is the weight array of the OUTPUT layer
    def concat_rweights(self, net, in_layer, d, out_layer, concat_dict, off=0):
        """Return weight genes with offsets already adjusted for randomized
        network.
        """
        #current layer is a concat layer; now we need to check *its* inputs
        #for i in  [list of input layer indexes]
        for i in (list(net._bottom_ids(list(net._layer_names).index(
            in_layer)))):
            ins = list(net._blob_names)[i]
            #assuming concats should no longer end up stacked
            off = self.create_rweights(ins, d, out_layer, net, off)
        return off

    #helper function for rand_weight_genes
    def create_rweights(self, in_layer, d, out_layer, net, off=0):
        """Create all weight genes in a random network and add them to
        weight chromosomes, and return new offset number.

        d: weight array of the output layer.
        """
        limit = net.blobs[in_layer].data.shape[1]
        new_off = off
        #i is the input number/node
        for i in range(limit):
            #j is the output number/node
            for j in range(len(d)):
                weight = d[j][i+off]
                w_gene = WeightGene(random.randint(1, 5), True, False,
                                    def_mut_rate, gene_ident(), weight,
                                    i, j, in_layer, out_layer)
                self.weightchr_a.append(w_gene)
                w_gene.dom = random.randint(1, 5)
                self.weightchr_b.append(w_gene)
            new_off += 1
        return new_off

    def mutate(self):
        """Handle mutation checks for all genes.
        """
        for layer in self.layerchr_a:
            result = layer.mutate()
            if result != "":
                self.handle_mutation(result, layer, "a", self.layerchr_a)
        for layer in self.layerchr_b:
            result = layer.mutate()
            if result != "":
                self.handle_mutation(result, layer, "b", self.layerchr_b)
        for weight in self.weightchr_a:
            result = weight.mutate()
            if result != "":
                self.handle_mutation(result, weight, "a", self.weightchr_a)
        for weight in self.weightchr_b:
            result = weight.mutate()
            if result != "":
                self.handle_mutation(result, weight, "b", self.weightchr_b)

    #helper function for mutate 
    def handle_mutation(self, result, gene, c, chro=None):
        """Handle changing a gene that has been mutated.
        """
        #this could be more complicated to take into account whether
        #the mutation actually changes anything, but keeping it simple for now
        #c is whether the chromosome is a or b, used for the mutaiton record
        if record_muts:
            self.mut_record.append([c, gene.ident, result])
        val = result[(result.index(",") + 2)::]
        #validation of this change is done at the mutate() function
        if result[0:3] == "Rat":
            val = float(val)
            gene.mut_rate += val
        elif result[0:3] == "Wei":
            val = float(val)
            gene.weight += val
        #validation of this change is done at the mutate() function
        elif result[0:3] == "Dom":
            val = int(val)
            gene.dom += val
        elif result[0:3] == "Dup":
            self.handle_duplication(gene, chro)
        elif result[0:3] == "Nod":
            if result[7] == "-":
                gene.nodes -= int(result[8])
            else:
                n_in = 0
                for g in chro:
                    if g.ident in gene.inputs:
                        n_in += g.nodes
                self.add_nodes(gene, chro, int(result[7]), self.weightchr_a,
                               n_in)
                self.add_nodes(gene, chro, int(result[7]), self.weightchr_b,
                               n_in)
                gene.nodes += int(result[7])

    #helper function for handle_mutation
    def handle_duplication(self, gene, chro):
        """Handle duplication mutations.
        """
        #first we make a new ident
        new_id = gene_ident()
        #then copy the gene
        new_gene = copy.deepcopy(gene)
        new_gene.ident = new_id
        #find index of gene, then stick new one in before that
        chro.insert(chro.index(gene), new_gene)
        #find a later gene that will use the new one as input
        out_gene = self.new_input(new_gene, chro)
        out_gene.inputs.append(new_gene.ident)
        #then make the new weight genes
        self.dup_weights(new_gene, out_gene, chro)

    #helper function for handle_duplication
    def new_input(self, gene, chro):
        """Return an existing gene that will take a new gene as input.
        Because of how Caffe works, this an only be a gene that comes
        after the new one.
        """
        potential = []
        for g in chro[(chro.index(gene)+1):]:
            if g.layer_type == "IP":
                potential.append(g)
        return(potential[random.randint(0, len(potential)-1)])

    #helper function for handle_duplication
    def dup_weights(self, new_gene, out_gene, chro):
        """Create the relevant new weight genes for a duplicated layer.
        """
        new_weights = []
        inputs = 0
        in_dict = {}
        out_inputs = new_gene.nodes
        #first we need to make weights from input -> new gene
        for g in chro:
            if g.ident in new_gene.inputs:
                inputs += g.nodes
                in_dict[g.ident] = g.nodes
            if g.ident in out_gene.inputs:
                out_inputs += g.nodes
        #xavier weight initialization: mean = 0 variance=1/n inputs
        #gaussian distribution
        var = 1/inputs
        #we give this function the std, not var
        var = math.sqrt(var)
        for layer in in_dict:
            for i in range(in_dict[layer]):
                for j in range(new_gene.nodes):
                    weight = random.gauss(0, var)
                    w = WeightGene(random.randint(1,5),
                                   True, False, def_mut_rate, gene_ident(),
                                   weight, i, j, layer, new_gene.ident)
                    new_weights.append(w)
        #then from new gene -> the gene that now takes it as input
        var = 1/out_inputs
        var = math.sqrt(var)
        for i in range(new_gene.nodes):
            for j in range(out_gene.nodes):
                weight = random.gauss(0, var)
                w = WeightGene(random.randint(1,5),
                               True, False, def_mut_rate,
                               gene_ident(), weight, i, j,
                               new_gene.ident, out_gene.ident)
                new_weights.append(w)
        #and lastly stick it all in the weight chromosome
        ###where?
        ###at end of both for now
        for gene in new_weights:
            self.weightchr_a.append(gene)
            self.weightchr_b.append(gene)

    #helper function for add_nodes
    #TODO: can I simplify this with the functions I just learned about?
    def find_n_inputs(self, gene, chro):
        """Return the total number of input layers to a layer gene
        and a dict of how many nodes each layer has.
        """
        inputs = 0
        in_dict = {}
        for g in chro[:(chro.index(gene))]:
            if g.ident in gene.inputs:
                inputs += g.nodes
                in_dict[g.ident] = g.nodes
        return inputs, in_dict
                               
    #helper function for handle_mutation
    def add_nodes(self, gene, chro, new_nodes, weight_chr, n_in):
        """Create new weight genes when nodes are added to a layer. Only
        works on one weight chromosome.
        """
        #the way this is written is intended to deal with two things
        #(1) edge case where nodes were larger in past, reduced, and are now
        #expanded again (e.g. 4 nodes -> 2 nodes -> adding two nodes)
        #trying to prevent duplicate weight genes
        #(2) keep weight genes nicely sorted by input node #
        done_ins = {}
        done_outs = []
        outs = self.find_outputs(gene, chro)
        out_dict = {}
        for layer in outs:
            out_dict[layer.ident] = layer
        for g in reversed(weight_chr):
            #outputs section
            #(where gene is the input for a given weight gene
            # and other layers are output)
            if g.in_layer == gene.ident and g.out_layer not in done_outs:
                new = (gene.nodes + new_nodes - 1) - g.in_node
                if new > 0:
                    ind = weight_chr.index(g) + 1
                    out_n = self.find_n_inputs(out_dict[g.out_layer],
                                                      chro)[0]
                    var = out_n/1
                    #we give this function the std, not var
                    var = math.sqrt(var)
                    for i in range(new):
                        for j in range(out_dict[g.out_layer].nodes):
                            w = random.gauss(0, var)
                            weight = WeightGene(random.randint(1, 5), True,
                                                False, def_mut_rate,
                                                gene_ident(),
                                                w, (i + g.in_node + 1),
                                                j, gene.ident,
                                                g.out_layer)
                            weight_chr.insert(ind, weight)
                            ind += 1
                done_outs.append(g.out_layer)
            #inputs section
            elif g.out_layer == gene.ident:
                if (g.in_layer in done_ins.keys() and
                    g.in_node in done_ins[g.in_layer]):
                    pass
                else:
                    new = (gene.nodes + new_nodes - 1) - g.out_node
                    if new > 0:
                        ind = weight_chr.index(g) + 1
                        var = n_in/1
                        var = math.sqrt(var)
                        for i in range(new):
                            w = random.gauss(0, var)
                            weight = WeightGene(random.randint(1, 5), True,
                                                False, def_mut_rate,
                                                gene_ident(), w,
                                                g.in_node, (i + g.out_node + 1),
                                                g.in_layer, gene.ident)
                            weight_chr.insert(ind, weight)
                            ind += 1
                    if g.in_layer in done_ins.keys():
                        done_ins[g.in_layer].append(g.in_node)
                    else:
                        done_ins[g.in_layer] = [g.in_node]

    #helper function for add_nodes
    #TODO: can I simplify this with the functions I just learned about?
    def find_outputs(self, gene, chro):
        """Return a list of layers that a layer gene outputs to. Used when
        adding weights so that concat layers are traced to their outputs.
        """
        out_list = []
        concats = []
        for g in chro:
            if gene.ident in g.inputs:
                if g.layer_type != "concat":
                    out_list.append(g)
                else:
                    concats.append(g.ident)
            for c in concats:
                if c in g.inputs:
                    out_list.append(g)
        return out_list

def network_ident():
    """Return a string that becomes a network's unique ID.
    """
    ident = ""
    while len(ident) != 11:
        if len(ident) == 3 or len(ident) == 7:
            ident = ident + "-"
        else:
            ident = ident + str(random.randint(0, 9))
    ident = "T" + ident
    return ident
             
class Gene(object):
    """Unit of information that populates a genome chromosome.
    
    dom: dominance rating (int, 1~5)
    can_mut: can it mutate? (bool)
    can_dup: can it be duplicated? (bool)
    mut_rate: mutation rate (float)
    ident: gene ID (str)
    """

    def __init__(self, dom, can_mut, can_dup, mut_rate, ident):
        self.dom = dom
        self.can_mut = can_mut
        self.can_dup = can_dup
        self.mut_rate = mut_rate
        self.ident = ident

    #defined in subclasses
    def mutate(self): # pragma: no cover
        """Return if and how a gene mutates.
        """
        raise NotImplementedError

    #defined in subclasses
    def read(self, active_list, other_gene, read_file): # pragma: no cover
        """Return which of a pair of genes (or what median result)
        is used to create the final network.

        active_list: a dict of layers used to build network: # nodes in layer.
        other_gene: equivalent gene on other chromosome.
        read_file: .gen file that is printed to, then read to make net.
        """
        raise NotImplementedError
        
    
def gene_ident():
    """Generate a six-character alphabetical string to use as
    a gene identifier.
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ident = ""
    while len(ident) < 6:
        ident = ident + letters[(random.randint(0, 25))]
    return ident

class LayerGene(Gene):
    """Defines a Caffe network layer.

    inputs: a list of strings that are inputs to this layer.
    nodes: an int (number of nodes this layer has) OR None )for concat layers).
    layer_type: a string defining the layer type
    """
    
    def __init__(self, dom, can_mut, can_dup, mut_rate, ident, inputs, nodes,
                 layer_type):
        super(LayerGene, self).__init__(dom, can_mut, can_dup, mut_rate, ident)
        self.inputs = inputs
        self.nodes = nodes
        self.layer_type = layer_type

    def read(self, active_list, other_gene, sub_dict, del_list):
        """Return either this gene or other_gene to be used in creating
        the network.

        Overrides read from base gene class. 
        """
        self_read = True
        other_read = True
        #first, make sure self or other isn't already in active_list
        #or we can get duplicates!
        if self.ident in active_list:
            self_read = False
        if other_gene.ident in active_list:
            other_read = False
        if self_read == True or other_read == True:
        #then, check if self's inputs are all in active_list
            if self.inputs == []:
                self_read = True
            else:
                if self_read == True:
                    ins = copy.copy(self.inputs)
                    for lay in ins:
                        if lay in del_list:
                            ins.remove(lay)
                        else:
                            if lay in sub_dict.keys():
                                lay = sub_dict[lay]
                            if lay not in active_list:
                                self_read = False
            if other_gene.inputs == []:
                other_read = True
            else:
                if other_read == True:
                    ins = copy.copy(other_gene.inputs)
                    for lay in ins:
                        if lay in del_list:
                            ins.remove(lay)
                        else:
                            if lay in sub_dict.keys():
                                lay = sub_dict[lay]
                            if lay not in active_list:
                                other_read = False
        #if only one can be read, read that
        if self_read == False:
            if other_read == False:
                return None
            else:
                return other_gene
        else:
            if other_read == False:
                return self
        #else, check dominance
        #if tie, flip a coin (layers do not co-dominate)
            else:
                if self.dom > other_gene.dom:
                    return self
                elif other_gene.dom > self.dom:
                    return other_gene
                else:
                    return random.choice([self, other_gene])
                                         
    #concat_dict entries format:
    #concat: [[in_layer1.ident, in_layer2.ident...][in_layer1.nodes,
    #       in_layer2.nodes...][out_layer1.ident, out_layer2.ident...]]
    def read_out(self, concat_dict, active_list):
        """Return a string with the layer parameters for the caffe file,
        including any necessary concat layers.
        """
        #if more than one input, need concat layers
        if len(self.inputs) > 1:
            in_con = None
            for key in concat_dict:
                #checking to see if any of the current concat layers
                #contain all of the inputs for this layer
                #(so we don't have to create a new one)
                #TODO: do we need to sort these?
                if self.inputs == concat_dict[key][0]:
                #if so, then add this layer as an output
                    concat_dict[key][2].append(self.ident)
                    in_con == key
                    break
                #else, need to make a new concat layer, and hence new entry
            if in_con == None:
                k = gene_ident()
                in_con = k
                in_nodes = []
                for lay in self.inputs:
                    in_nodes.append(active_list[lay])
                concat_dict[k] = [self.inputs, in_nodes, [self.ident]]
            #print the concat layer first
            result = dedent(layer_dict["concat_0"].format(k)) + "\n"
            for lay in self.inputs:
                result += "  bottom: \"" + lay + "\"\n"
            result += dedent(layer_dict["concat_1"].format(k))
            x = self.inputs
            #now this layer
            self.inputs = [in_con]
            result += dedent(layer_dict[self.layer_type].format(self))
            self.inputs = x
        else:
            result = dedent(layer_dict[self.layer_type].format(self))
        return result

    def mutate(self):
        """Return a string with whether and how the gene mutates.

        Overrides mutate from base gene class.
        """
        if not self.can_mut:
            return ""
        else:
            roll = random.random()
            if roll > self.mut_rate:
                return ""
            else:
                result = self.determine_mutation()
                return result

    def determine_mutation(self):
        """Return the result from a mutation event.
        """
        roll = random.random()
        #mutate dom
        if roll < layer_mut_probs[0]:
            change = 0
            while change == 0:
                change = int(random.gauss(0, 1))
                while change + self.dom > 5:
                    change -= 1
                while change + self.dom < 1:
                    change += 1
            result = "Dom, " + str(change)
        #mutate rate
        elif roll > layer_mut_probs[0] and roll < sum(layer_mut_probs[0:2]):
            change = 0
            while change == 0:
                change = random.gauss(0, sigma)
                while change + self.mut_rate > 1 or change + self.mut_rate <= 0:
                    change = random.gauss(0, sigma)
            result = "Rate, " + str(change)
        #mutate num
        elif (roll > sum(layer_mut_probs[0:2])
              and roll < sum(layer_mut_probs[0:3])):
            change = 0
            while change == 0:
                change = int(random.gauss(0, 1))
                while change + self.nodes < 1:
                    change += 1
            result = "Nodes, " + str(change)
        #dup
        elif (roll > sum(layer_mut_probs[0:3]) and
              roll < sum(layer_mut_probs[0:4])):
            result = "Duplicate,"
        #add input
        else:
            result = "Add input,"
        return result
    
class WeightGene(Gene):
    """Defines a single weight in the Caffe network.

    weight: weight value that this gene codes for (float)
    in/out_node: input node from input layer, output node in output layer (int)
    in/out_layer: input/output layer ID (str)
    """
    
    def __init__(self, dom, can_mut, can_dup, mut_rate, ident, weight, in_node,
                 out_node, in_layer, out_layer):
        super(WeightGene, self).__init__(dom, can_mut, can_dup,
                                          mut_rate, ident)
        self.weight = weight
        self.in_node = in_node
        self.out_node = out_node
        self.in_layer = in_layer
        self.out_layer = out_layer
        #this is here to deal with concat layers
        self.alt_in = in_node

    def read(self, active_list, sub_dict, other_gene=None):
        """Return either this gene or other_gene, or an average weight
        if they co-dominate, to be used in creating the network.

        Overrides read from base gene class. 
        """
        #first check that input and output are in dict (inc. node #) for both
        #if both there, but if only one can be read, read that
        #first check if they can be read
        self_read = self.can_read(active_list, sub_dict)
        if other_gene is not None:
            other_read = other_gene.can_read(active_list, sub_dict)
        else:
            other_read = False
        if self_read == False:
            if other_read == False:
                return None
            if other_gene.in_layer in sub_dict:
                in_lay = sub_dict[other_gene.in_layer]
            else:
                in_lay = other_gene.in_layer
            if other_gene.out_layer in sub_dict:
                out_lay = sub_dict[other_gene.out_layer]
            else:
                out_lay = other_gene.out_layer
            return [in_lay, other_gene.alt_in, out_lay,
                    other_gene.out_node, other_gene.weight]
        else:
            if self.in_layer in sub_dict:
                s_in_lay = sub_dict[self.in_layer]
            else:
                s_in_lay = self.in_layer
            if self.out_layer in sub_dict:
                s_out_lay = sub_dict[self.out_layer]
            else:
                s_out_lay = self.out_layer
            if other_read == False:
                return [s_in_lay, self.alt_in, s_out_lay, self.out_node,
                        self.weight]
            #if both can be read AND they are NOT for the same input/output,
            #read both
            else:
                if other_gene.in_layer in sub_dict:
                    o_in_lay = sub_dict[other_gene.in_layer]
                else:
                    o_in_lay = other_gene.in_layer
                if other_gene.out_layer in sub_dict:
                    o_out_lay = sub_dict[other_gene.out_layer]
                else:
                    o_out_lay = other_gene.out_layer
                if (s_in_lay == o_in_lay and
                    s_out_lay == o_out_lay):
                    #else, if both read AND they are for the same input/output,
                    #check dom
                    #if same, can co-dominate (average values)
                    if (self.in_node == other_gene.in_node
                        and self.out_node == other_gene.out_node):
                        if self.dom < other_gene.dom:
                            return [o_in_lay, other_gene.alt_in,
                                    o_out_lay, other_gene.out_node,
                                    other_gene.weight]
                        elif self.dom > other_gene.dom:
                            return [s_in_lay, self.alt_in, s_out_lay,
                                    self.out_node, self.weight]
                        else:
                            return [s_in_lay, self.alt_in, s_out_lay,
                                    self.out_node, (self.weight +
                                                    other_gene.weight)/2]
                #reading both if not same
                return [s_in_lay, self.alt_in, s_out_lay, self.out_node,
                        self.weight, o_in_lay, other_gene.alt_in,
                        o_out_lay, other_gene.out_node, other_gene.weight]

    def can_read(self, active_list, sub_dict):
        """Return whether this gene defines an existing weight in the network.
        """
        #check if input/output layers exist
        if self.in_layer in sub_dict:
            in_layer = sub_dict[self.in_layer]
        else:
            in_layer = self.in_layer
        if self.out_layer in sub_dict:
            out_layer = sub_dict[self.out_layer]
        else:
            out_layer = self.out_layer
        if in_layer in active_list:
            if out_layer in active_list:
                #now check in/out nodes
                if active_list[in_layer] >= (self.in_node + 1):
                    if active_list[out_layer] >= (self.out_node + 1):
                        return True
        return False

    def mutate(self):
        """Return a string with whether and how the gene mutates.

        Overrides mutate from base gene class.
        """
        if not self.can_mut:
            return ""
        else:
            roll = random.random()
            if roll > self.mut_rate:
                return ""
            else:
                result = self.determine_mutation()
                return result

    def determine_mutation(self):
        """Return the result from a mutation event.
        """
        roll = random.random()
        if roll < weight_mut_probs[0]:
            #change weight
            change = random.gauss(0, 0.50)
            result = "Weight, " + str(change)
        elif roll > weight_mut_probs[0] and roll < sum(weight_mut_probs[0:2]):
            #change dom
            change = 0
            while change == 0:
                change = int(random.gauss(0, 1))
                while change + self.dom > 5:
                    change -= 1
                while change + self.dom < 1:
                    change += 1
            result = "Dom, " + str(change)
        else:
            #change mutation rate
            change = 0
            while change == 0:
                change = random.gauss(0, sigma)
                while change + self.mut_rate > 1 or change + self.mut_rate <= 0:
                    change = random.gauss(0, sigma)
            result = "Rate, " + str(change)
        return result


class HaploidGenome(Genome):
    """Haploid genome defining a neural network.

    The second layer/weight chromosomes are left blank.
    """

    def __init__(self, layerchr, weightchr):
        super().__init__(layerchr, [], weightchr, [])

    def recombine(self, other_genome):
        """Return a new child genome from two parent genomes.

        Overrides recombine from base genome class. 
        """
        #first, do 'crossover' b/w the two genomes
        #hm... could we cheat here real quick:
        self.layerchr_b = other_genome.layerchr_a
        self.weightchr_b = other_genome.weightchr_a
        result = self.crossover()
        self.layerchr_b = []
        self.weightchr_b = []
        #then randomly pick one possible child
        layers = random.sample([result.layerchr_a, result.layerchr_b], 1)
        weights = random.sample([result.weightchr_a, result.weightchr_b], 1)
        layers = layers[0]
        weights = weights[0]
        child = HaploidGenome(layers, weights)
        child.mutate()
        return child
