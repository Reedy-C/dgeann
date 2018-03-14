import random
from textwrap import dedent
import os
import math
import caffe
import numpy

#default solver
solv = '''\
        net: "{0}"
        solver_type: ADADELTA
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

#dict for layer types, used to generate caffe network def files
layer_dict = {"input":'''\
                        input: "{0.ident}"
                        input_shape: {{
                          dim: 1
                          dim: 1
                          dim: 1
                          dim: {0.nodes}
                        }}
                        ''',
                "concat": '''\
                            layer {{
                              name: "{0.ident}"
                              type: "Concat"
                              bottom: "{0.inputs[0]}"
                              bottom: "{0.inputs[1]}"
                              top: "{0.ident}"
                              concat_param {{
                                axis: 3
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
#this can be changed, but numbers should add up to ~1
#(or mutations will not happen correctly)
#TODO - check that?
#current probabilities were basically made up on the spot
layer_mut_probs = (0.333, 0.234, 0.167, 0.133, 0.133)

#same as above, but for weight gene mutations
#1 is probability of a weight mutation
#2 is probability of a dominance mutation
#3 is probability of a mutation rate mutation
weight_mut_probs = (0.833, 0.117, 0.05)

        
class genome(object):

    #a genome is a list of lists of genes
    #referred to as chromosomes
    #these genomes are diploid, so two copies of each chromosome
    #(which may have different genes on them)
    #chromosome 1: layer genes
    #chromosome 2: weight genes
    def __init__(self, layerchr_a, layerchr_b, weightchr_a, weightchr_b):
        self.layerchr_a = layerchr_a
        self.layerchr_b = layerchr_b
        self.weightchr_a = weightchr_a
        self.weightchr_b = weightchr_b

    #recombine takes two genomes (this and one other)
    #and produces a new mixed-up child genome
    def recombine(self, other_genome):
        #first we need to do crossover on each genome
        parent_one = self.crossover()
        parent_two = other_genome.crossover()
        #then randomly pick chromosome for layers and weights from each
        layer_one = random.sample([parent_one.layerchr_a, parent_two.layerchr_a], 1)
        layer_two = random.sample([parent_one.layerchr_b, parent_two.layerchr_b], 1)
        weight_one = random.sample([parent_one.weightchr_a, parent_two.weightchr_a], 1)
        weight_two = random.sample([parent_one.weightchr_b, parent_two.weightchr_b], 1)
        layer_one = layer_one[0]
        layer_two = layer_two[0]
        weight_one = weight_one[0]
        weight_two = weight_two[0]
        child = genome(layer_one, layer_two, weight_one, weight_two)
        #now just do mutations
        child.mutate()
        return child

    #crosses over every pair of chromosomes
    #returns a new genome
    def crossover(self):
        n, m = self.last_shared()
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
        weight_cross = random.randint(1, m)
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
        result = genome(layer_a, layer_b, weight_a, weight_b)
        return result

    #helper function for crossover
    #it finds the last possible crossover points
    #trying to minimize the wrecking of future evolved structures here
    #hence why it's not just randint(0, len(chromosome)-1)
    def last_shared(self):
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
                if (n+1) < len(self.layerchr_a) and (n+1) < len(self.layerchr_b):
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

    #build a new network from the genome
    #delete: if true, deletes generated network files
    #set to False if you want eg a record of how the network structure
    #changed network-by-network
    #returns the solver that contains the network built from genome
    #TODO is it possible to simplify and get rid of active_list
    #   given that I now know that list(t._layer/blob_names) exists?
    def build(self, delete=True):
        #first, generate a new ID for the network
        self.ident = genome.generate_ident()
        if delete == False:
            if not os.path.exists('Gen files'):
                os.makedirs('Gen files')
            ident_file = os.path.join('Gen files', self.ident)
        else:
            ident_file = self.ident
        ident_file = ident_file + '.gen'
        active_list = {}
        concat_dict = {}
        if len(self.layerchr_a) != len(self.layerchr_b):
            #if the chromosomes are not equal in length, read both
            #for the length of the shorter one
            #then read the rest on the longer one
            if len(self.layerchr_a) > len(self.layerchr_b):
                for n in range(len(self.layerchr_b)):
                    active_list = self.layerchr_a[n].read(ident_file, active_list,
                                                     concat_dict, self.layerchr_b[n])
                x = len(self.layerchr_b)
                while x < len(self.layerchr_a):
                    active_list = self.layerchr_a[x].read(ident_file, active_list,
                                                     concat_dict, None)
                    x += 1
            else:
                for n in range(len(self.layerchr_a)):
                    active_list = self.layerchr_b[n].read(ident_file, active_list,
                                                     concat_dict, self.layerchr_a[n])
                x = len(self.layerchr_a)
                while x < len(self.layerchr_b):
                    active_list = self.layerchr_b[x].read(ident_file, active_list,
                                                     concat_dict, None)
                    x += 1
        else:
            #if they're the same length, hooray! just read them
            for n in range(len(self.layerchr_a)):
                active_list = self.layerchr_a[n].read(ident_file, active_list,
                                                      concat_dict, self.layerchr_b[n])
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
        self.concats(active_list, concat_dict)
        #now change the weights to those specified in genetics
        if len(self.weightchr_a) > 0:
            self.build_weights(active_list, solver.net)
        else:
            self.rand_weight_genes(solver.net, concat_dict)
        return solver

    #returns a string that becomes a network's unique ID
    @staticmethod
    def generate_ident():
        ident = ""
        while len(ident) != 11:
            if len(ident) == 3 or len(ident) == 7:
                ident = ident + "-"
            else:
                ident = ident + str(random.randint(0, 9))
        ident = "T" + ident
        return ident

    #this is here to deal with what happens when we have concat layers
    #if a and b are concatted and fed to c, a is okay
    #but any b->c weight genes won't work right
    #since 0 in b is now offset by len(a)
    #(for the purposes of weights at least)
    #easiest option seems to be 'adjust the inputs'?
    #also needs to handle d->c,e->c,f->c, a->g...
    def concats(self, active_list, concat_dict):
        for weight in self.weightchr_a:
            if weight.in_layer in concat_dict:
                if weight.out_layer in concat_dict[weight.in_layer][2::]:
                    n = concat_dict[weight.in_layer][1]
                    weight.alt_in += n

    #reads the chromosomes that specify the network weights
    #and changes the weights in the created network to those weights
    #takes the list of layers and a network
    def build_weights(self, active_list, net):
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
                    values = a.read(active_list, b)
                    if values is not None:
                        genome.adjust_weight(net, values)
                    n += 1
                    m += 1
                    if n > a_lim and m <= b_lim:
                        longer = weightchr_b
                    elif m > b_lim and n <= a_lim:
                        longer = weightchr_a
                else:
                    #currently not sure how we could get here
                    #but might as well have a plan if we do
                    if a.in_node == 0 and a.out_node == 0:
                        while b.out_node != 0:
                            m, b = self.read_through("b", m, active_list, net)
                            if m > b_lim:
                                longer = self.weightchr_a
                                break
                    elif b.in_node == 0 and b.out_node == 0:
                        while a.out_node != 0:
                            n, a = self.read_through("a", n, active_list, net)
                            if n > a_lim:
                                longer = self.weightchr_b
                                break
                    else:
                        while a.out_node < b.out_node:
                            n, a = self.read_through("a", n, active_list, net)
                            if n > a_lim:
                                longer = self.weightchr_b
                                break
                            if a.in_node == 0 and a.out_node == 0:
                                break
                        while b.out_node < a.out_node:
                            m, b = self.read_through("b", m, active_list, net)
                            if m > b_lim:
                                longer = self.weightchr_a
                                break
                            if b.in_node == 0 and b.out_node == 0:
                                break
            else:
                #if one is at 0 input/0 output, read the other 'till it catches up
                if a.in_node == 0 and a.out_node == 0:
                    while b.in_node != 0:
                        m, b = self.read_through("b", m, active_list, net)
                        if m > b_lim:
                            break
                    if m <= b_lim:
                        while b.out_node != 0:
                            m, b = self.read_through("b", m, active_list, net)
                            if m > b_lim:
                                break
                elif b.in_node == 0 and b.out_node == 0:
                    while a.in_node != 0:
                        n, a = self.read_through("a", n, active_list, net)
                        if n > a_lim:
                            break
                    if n <= a_lim:
                        while a.out_node != 0:
                            n, a = self.read_through("a", n, active_list, net)
                            if n > a_lim:
                                break
                #if one is at a higher input #, read the other 'till it catches up
                #or until the other hits 0/0 i/o
                else:
                    while a.in_node < b.in_node:
                        n, a = self.read_through("a", n, active_list, net)
                        if n > a_lim:
                            longer = self.weightchr_b
                            break
                        if a.in_node == 0 and a.out_node == 0:
                            break
                    while b.in_node < a.in_node:
                        m, b = self.read_through("b", m, active_list, net)
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
                    values = longer[x].read(active_list)
                    if values is not None:
                        genome.adjust_weight(net, values)
                    x += 1

    #helper function for build_weights
    #takes the caffe network
    #and the returned strings from reading a weight gene
    #adjusts the corresponding weight in the net to the specified weight
    @staticmethod
    def adjust_weight(net, values):
        #values is a list formatted as:
        #input (str), in node, output (str), out node, weight
        #with perhaps another set for a second weight adjustment
        #TODO no use for input str?
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
    #takes chromosome, n (position of gene in chromosome)
    #active_list
    #net
    #advances down one chromosome at a time
    def read_through(self, chro, n, active_list, net):
        if chro == "a":
            chro = self.weightchr_a
        elif chro == "b":
            chro = self.weightchr_b
        a = chro[n]
        values = a.read(active_list)
        if values is not None:
            genome.adjust_weight(net, values)
        n += 1
        if n <= (len(chro) - 1):
            a = chro[n]
        return n, a

    #helper function for build
    #starts with a random network
    #turns the random weights into weight genes that are added to both
    #chromosomes and given random dominance
    #(can be different for either copy on either chromosome)
    #TODO return?
    def rand_weight_genes(self, net, concat_dict):
        for key in net.params:
            d = net.params[key][0].data
            if type(d[0]) == numpy.ndarray:
                #have to find in_layer, undo concats
                in_layer = list(net._blob_names)[
                    list(net._bottom_ids(list(net._layer_names).index(key)))[0]]
                #now see if this is in concat_dict?
                conc = False
                for key in concat_dict:
                    if concat_dict[key][0] == in_layer:
                        #then input is from a concat layer
                        conc = True
                        break
                if conc == True:
                    self.concat_rweights(net, in_layer, d, key, concat_dict)
                else:
                    self.create_rweights(in_layer, d, key, net)

    #helper function for rand_weight_genes
    #helps sort out concats... again
    #t is net
    #d is the weight array of the OUTPUT layer
    #TODO more elegant way to do this?
    def concat_rweights(self, net, in_layer, d, out_layer, concat_dict, off=0):
        #current layer is a concat layer; now we need to check *its* inputs
        ins_left = list(net._blob_names)[
                    list(net._bottom_ids(
                        list(net._layer_names).index(in_layer)))[0]]
        ins_right = list(net._blob_names)[
                    list(net._bottom_ids(
                        list(net._layer_names).index(in_layer)))[1]]
        #check if left is a concat
        left_concat = False
        right_concat = False
        for key in concat_dict:
            if concat_dict[key][0] == ins_left:
                left_concat = True
            elif concat_dict[key][0] == ins_right:
                right_concat = True
            if left_concat == True and right_concat == True:
                break
        #if it is, then repeat
        if left_concat == True:
            off = self.concat_rweights(net, ins_left, d, out_layer,
                                       concat_dict, off)
        #else, create genes and return offset
        else:
            off = self.create_rweights(ins_left, d, out_layer, net, off)
        if right_concat == True:
            off = self.concat_rweights(net, ins_right, d, out_layer,
                                       concat_dict, off)
        else:
            off = self.create_rweights(ins_right, d, out_layer, net, off)
        return off

    #helper function for rand_weight_genes
    #creates and appends all weight genes
    #d is the weight array of the OUTPUT layer
    def create_rweights(self, in_layer, d, out_layer, net, off=0):
        limit = net.blobs[in_layer].data.shape[3]
        new_off = off
        #i is the output number/node
        for i in range(len(d)):
            #j is the input number/node
            for j in range(limit):
                weight = d[i][j+off]
                w_gene = weight_gene(random.randint(1, 5), True, False,
                                     def_mut_rate, self.new_ident(), weight,
                                     i, j, in_layer, out_layer)
                self.weightchr_a.append(w_gene)
                w_gene.dom = random.randint(1, 5)
                self.weightchr_b.append(w_gene)
            new_off += 1
        return new_off

    #run through all genes to see if any mutate
    def mutate(self):
        for layer in self.layerchr_a:
            result = layer.mutate()
            if result is not "":
                self.handle_mutation(result, layer)
        for layer in self.layerchr_b:
            result = layer.mutate()
            if result is not "":
                self.handle_mutation(result, layer)
        for weight in self.weightchr_a:
            result = weight.mutate()
            if result is not "":
                self.handle_mutation(result, weight)
        for weight in self.weightchr_b:
            result = weight.mutate()
            if result is not "":
                self.handle_mutation(result, weight)

    #implements changes to genes from mutations
    def handle_mutation(self, result, gene, chro=None):
        val = result[(result.index(",") + 2)::]
        if result[0:3] == "Rat":
            val = float(val)
            gene.mut_rate += val
            if gene.mut_rate > 1:
                gene.mut_rate = 1
            elif gene.mut_rate < 0:
                gene.mut_rate = 0
        elif result[0:3] == "Wei":
            val = float(val)
            gene.weight += val
        elif result[0:3] == "Dom":
            val = int(val)
            gene.dom += val
            if gene.dom > 5:
                gene.dom = 5
            elif gene.dom < 0:
                gene.dom = 0
        elif result[0:3] == "Dup":
            self.handle_duplication(gene, chro)
        elif result[0:3] == "Nod":
            if result[7] == "-":
                gene.nodes -= int(result[8])
            else:
                n_in, d = self.find_n_inputs(gene, chro)
                self.add_nodes(gene, chro, int(result[7]), self.weightchr_a, n_in, d)
                self.add_nodes(gene, chro, int(result[7]), self.weightchr_b, n_in, d)
                gene.nodes += int(result[7])
                
    #generates a six-character alphabetical string to use as a gene identifier
    #TODO move?
    @staticmethod
    def new_ident():
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ident = ""
        while len(ident) < 6:
            ident = ident + letters[(random.randint(0, 25))]
        return ident

    #helper function for handle_mutation
    #it handles duplication mutations
    def handle_duplication(self, gene, chro):
        #first we make a new ident
        new_id = genome.new_ident()
        #then copy the gene
        new_gene = layer_gene(gene.dom, gene.can_mut, gene.can_dup, gene.mut_rate,
                              new_id, gene.inputs, gene.nodes, gene.layer_type)
        #find index of gene, then stick new one in before that
        chro.insert(chro.index(gene), new_gene)
        #find a later gene that will use the new one as input
        out_gene = self.new_input(new_gene, chro)
        #add concats
        self.add_concats(new_gene, out_gene, chro)
        #then make the new weight genes
        self.dup_weights(new_gene, out_gene, chro)

    #helper function for handle_duplication
    #finds a gene that can use a new gene as input
    #because of how caffe works, for the sake of simplicity in genome reading
    #this can only be a gene that comes _after_ the new gene
    def new_input(self, gene, chro):
        potential = []
        for g in chro[(chro.index(gene)+1):]:
            if g.layer_type == "IP":
                potential.append(g)
        return(potential[random.randint(0, len(potential)-1)])

    #helper function for handle_duplication
    #when adding a new layer, it makes the relevant new weight genes
    def dup_weights(self, new_gene, out_gene, chro):
        new_weights = []
        inputs = 0
        in_dict = {}
        #first we need to make weights from input -> new gene
        for g in chro:
            if g.ident in new_gene.inputs:
                in_gene = g
                break
        if in_gene.layer_type == "concat":
            inputs, in_dict = self.find_n_genes(out_gene, chro)
        else:
            inputs = in_gene.nodes
            in_dict[in_gene.ident] = in_gene.nodes
        for layer in in_dict:
            #xavier weight initialization: mean = 0 variance=1/n inputs
            #gaussian distribution
            var = 1/inputs
            #we give this function the std, not var
            var = math.sqrt(var)
            weight = random.gauss(0, var)
            for i in range(in_dict[layer]):
                for j in range(new_gene.nodes):
                    w = weight_gene(random.randint(1,5),
                                    True, False, 3, genome.new_ident(),
                                    weight, i, j, layer, new_gene.ident)
                    new_weights.append(w)
        #then from new gene -> the gene that now takes it as input
        for i in range(new_gene.nodes):
            for j in range(out_gene.nodes):
                inputs, d = self.find_n_inputs(out_gene, chro)
                weight = 1/inputs
                w = weight_gene(random.randint(1,5),
                                True, False, 3, genome.new_ident(),
                                weight, i, j, new_gene.ident, out_gene.ident)
                new_weights.append(w)
        #and lastly stick it all in the weight chromosome
        ###where?
        ###at end of both for now
        for gene in new_weights:
            self.weightchr_a.append(gene)
            self.weightchr_b.append(gene)

    #helper function for dup_weights
    #returns total inputs to gene (adds together concat inputs)
    #also returns a dict (dict[layer_name] = layer_nodes)
    #TODO: can I simplify this with the functions I just learned about?
    def find_n_inputs(self, gene, chro):
        inputs = 0
        in_dict = {}
        for g in chro[:(chro.index(gene))]:
            if g.ident in gene.inputs:
                inp = g
                break
        if inp.layer_type == "concat":
            for g in chro[:(chro.index(inp))]:
                if g.ident in inp.inputs:
                    inputs += g.nodes
                    in_dict[g.ident] = g.nodes
        else:
            inputs = inp.nodes
            in_dict[inp.ident] = inp.nodes
        return inputs, in_dict

    #helper function for handle_duplication
    #deals with the fact that caffe IP layers can't take more than one input
    #and need them put into a concat layer, or if it already takes input from one
    #then we need to adjust *that* layer
    def add_concats(self, new_gene, out_gene, chro):
        #since in caffe an IP layer can't have more than one input
        #if adding an input, we need to change the input to a concat layer
        #...unless the input already is one
        #basically, we don't want to orphan any layers involved
        for gene in chro:
            conc = []
            if gene.ident == out_gene.inputs[0]:
                if gene.layer_type != "concat":
                    #make new concat layer
                    conc = layer_gene(random.randint(1,5), False, False, 0,
                                      genome.new_ident(),
                                      [gene.ident, new_gene.ident],
                                      None, "concat")
                else:
                    #in that case, we should see if anyone else is using it
                    #as an input
                    #so search the genes AFTER this one
                    i = chro.index(out_gene)
                    for g in chro[i:]:
                        if g.ident != out_gene.ident and out_gene.inputs[0] in g.inputs:
                            inputs = out_gene.inputs
                            inputs.append(new_gene.ident)
                            #if so, make a new concat with old + new inputs
                            conc = layer_gene(random.randint(1,5),
                                              False, False, 0,
                                              genome.new_ident(), inputs,
                                              None, "concat")
                    if conc == []:
                    #and if not, move it w/ new gene added as input
                        for g in chro[:i]:
                            if g.ident == out_gene.inputs[0]:
                                conc = g
                        chro.remove(conc)
                        conc.inputs.append(new_gene.ident)
                        conc.ident = genome.new_ident()
            if conc != []:
                chro.insert(chro.index(out_gene), conc)
                out_gene.inputs = [conc.ident]
                break
                               
    #helper function for handle_mutation
    #when adding nodes to a layer, it deals with adding new weight genes
    #unlike other helper functions for handle_mutation, it works on only one
    #weight chromosome at a time and does not change the gene (handle_mutation does)
    def add_nodes(self, gene, chro, new_nodes, weight_chr, n_in, d):
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
                    out_n, out_d = self.find_n_inputs(out_dict[g.out_layer],
                                                      chro)
                    var = out_n/1
                    #we give this function the std, not var
                    var = math.sqrt(var)
                    for i in range(new):
                        for j in range(out_dict[g.out_layer].nodes):
                            w = random.gauss(0, var)
                            weight = weight_gene(random.randint(1, 5), True,
                                                 False, 3, genome.new_ident(),
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
                            weight = weight_gene(random.randint(1, 5), True,
                                                 False, 3, genome.new_ident(), w,
                                                 g.in_node, (i + g.out_node + 1),
                                                 g.in_layer, gene.ident)
                            weight_chr.insert(ind, weight)
                            ind += 1
                    if g.in_layer in done_ins.keys():
                        done_ins[g.in_layer].append(g.in_node)
                    else:
                        done_ins[g.in_layer] = [g.in_node]

    #helper function for add_nodes
    #returns a list of layers that gene outputs to
    #this is used for weights, so any concats are traced to their outputs
    #TODO: can I simplify this with the functions I just learned about?
    def find_outputs(self, gene, chro):
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
   
             
class gene(object):

    #dom = dominance rating (int, 1~5)
    #can_mut = can it mutate? (bool)
    #can_dup = can it be duplicated? (bool)
    #mut_rate = mutation rate (float)
    #ident = gene ID (str)
    def __init__(self, dom, can_mut, can_dup, mut_rate, ident):
        self.dom = dom
        self.can_mut = can_mut
        self.can_dup = can_dup
        self.mut_rate = mut_rate
        self.ident = ident

    #defined in subclasses
    def mutate(self):
        pass

    #defined in subclasses
    #active_list = a dict of layers used to build network: # nodes in layer
    #other_gene = equivalent gene on other chromosome,
    #               used to determine phenotype
    #read_file = .gen file that is printed to, later read into caffe to make net
    def read(self, active_list, other_gene, read_file):
        pass


class layer_gene(gene):

    #inputs is a list of strings that are inputs to this layer
    #nodes is an int (number of nodes this layer has OR None for concat layers)
    #layer type is a string
    def __init__(self, dom, can_mut, can_dup, mut_rate, ident, inputs, nodes,
                 layer_type):
        super(layer_gene, self).__init__(dom, can_mut, can_dup, mut_rate, ident)
        self.inputs = inputs
        self.nodes = nodes
        self.layer_type = layer_type

    def read(self, read_file, active_list, concat_dict, other_gene=None):
        ##first, check if self's inputs are all in active_list
        if self.inputs == []:
            self_read = True
        else:
            for layer in self.inputs:
                if layer not in active_list:
                    self_read = False
                else:
                    self_read = True
        if other_gene is None:
            other_read = False
        else:
            if other_gene.inputs == []:
                other_read = True
            else:
                for layer in other_gene.inputs:
                    if layer not in active_list:
                        other_read = False
                    else:
                        other_read = True
        ##if neither can be read, return unchanged active_list
        ##if only one, read that
        if self_read == False:
            if other_read == False:
                return active_list
            else:
                print_out = other_gene.read_out()
                result = active_list.copy()
                result[other_gene.ident] = other_gene.nodes
                other_gene.concats(concat_dict, active_list)
        else:
            if other_read == False:
                print_out = self.read_out()
                result = active_list.copy()
                result[self.ident] = self.nodes
                self.concats(concat_dict, active_list)
        ##else, check dominance
        ##if tie, flip a coin (layers do not co-dominate)
            else:
                result = active_list.copy()
                if self.dom > other_gene.dom:
                    print_out = self.read_out()
                    result[self.ident] = self.nodes
                    self.concats(concat_dict, active_list)
                elif other_gene.dom > self.dom:
                    print_out = other_gene.read_out()
                    result[other_gene.ident] = other_gene.nodes
                    other_gene.concats(concat_dict, active_list)
                else:
                    choice = random.randint(0, 1)
                    if choice == 0:
                        print_out = self.read_out()
                        result[self.ident] = self.nodes
                        self.concats(concat_dict, active_list)
                    else:
                        print_out = other_gene.read_out()
                        result[other_gene.ident] = other_gene.nodes
                        other_gene.concats(concat_dict, active_list)
        ##output relevant text to file and return changed active_list
        f = open(read_file, "a")
        f.write(print_out)
        f.close()
        return result

    #a function that produces the string a gene makes for the caffe file
    def read_out(self):
        result = dedent(layer_dict[self.layer_type].format(self))
        return result

    #helper function for read to help create concat layers
    #dict entries format:
    #'input 2.ident': ['concat_ident', input1.nodes,
    #               'output_ident', 'output_ident'...]
    def concats(self, concat_dict, active_list):
        if self.layer_type == "concat":
            len_a = active_list[self.inputs[0]]
            concat_dict[self.inputs[1]] = [self.ident, len_a]
        elif self.layer_type == "IP":
            for key in concat_dict:
                if concat_dict[key][0] == self.inputs[0]:
                    concat_dict[key].append(self.ident)

    def mutate(self):
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
                #TODO test
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
                while change + self.nodes < 0:
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
    
class weight_gene(gene):

    #weight = weight that this gene codes for (float)
    #in/out node = input node from input layer, output node in output layer (int)
    #in/out layer = input/output layer (str)
    def __init__(self, dom, can_mut, can_dup, mut_rate, ident, weight, in_node,
                 out_node, in_layer, out_layer):
        super(weight_gene, self).__init__(dom, can_mut, can_dup,
                                          mut_rate, ident)
        self.weight = weight
        self.in_node = in_node
        self.out_node = out_node
        self.in_layer = in_layer
        self.out_layer = out_layer
        #this is here to deal with concat layers
        self.alt_in = in_node

    def read(self, active_list, other_gene=None):
        #first check that input and output are in dict (inc. node #) for both
        #if both there, but if only one can be read, read that
        #first check if they can be read
        self_read = self.can_read(active_list)
        if other_gene is not None:
            other_read = other_gene.can_read(active_list)
        else:
            other_read = False
        if self_read == False:
            if other_read == False:
                return None
            return [other_gene.in_layer, other_gene.alt_in, other_gene.out_layer,
                    other_gene.out_node, other_gene.weight]
        else:
            if other_read == False:
                return [self.in_layer, self.alt_in, self.out_layer,
                        self.out_node, self.weight]
            #if both can be read AND they are NOT for the same input/output, read both
            else:
                if (self.in_layer == other_gene.in_layer and
                    self.out_layer == other_gene.out_layer):
                    #else, if both read AND they are for the same input/output, check dom
                    #if same, can co-dominate (average values)
                    if (self.in_node == other_gene.in_node
                        and self.out_node == other_gene.out_node):
                        if self.dom < other_gene.dom:
                            return [other_gene.in_layer, other_gene.alt_in,
                                    other_gene.out_layer, other_gene.out_node,
                                    other_gene.weight]
                        elif self.dom > other_gene.dom:
                            return [self.in_layer, self.alt_in, self.out_layer,
                                    self.out_node, self.weight]
                        else:
                            average = (self.weight + other_gene.weight)/2
                            return [self.in_layer, self.alt_in, self.out_layer,
                                    self.out_node, average]
                #reading both if not same
                return [self.in_layer, self.alt_in, self.out_layer, self.out_node,
                        self.weight, other_gene.in_layer, other_gene.alt_in,
                        other_gene.out_layer, other_gene.out_node, other_gene.weight]

    def can_read(self, active_list):
        result = False
        #check if input/output layers exist
        if self.in_layer in active_list:
            if self.out_layer in active_list:
                #now check in/out nodes
                if active_list[self.in_layer] >= (self.in_node + 1):
                    if active_list[self.out_layer] >= (self.out_node + 1):
                        result = True
        return result

    def mutate(self):
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
