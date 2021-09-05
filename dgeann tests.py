import os
from textwrap import dedent
import unittest

import caffe
import dgeann
from dgeann import WeightGene as wg
from dgeann import LayerGene as lg

#tests that weight genes read and mutate properly
class testWeight(unittest.TestCase):

    def setUp(self):
        self.active_list = {"A": 6, "B": 3}
        self.unread_a = wg(1, False, False, 0, "asld", 0.00, 0, 0, "n", "N")
        self.unread_b = wg(1, False, False, 0, "asld", 0.00, 6, 0, "A", "B")
        self.low_dom_a = wg(1, False, False, 0, "asld", 0.00, 3, 0, "A", "B")
        self.low_dom_b = wg(1, False, False, 0, "asld", 5.00, 3, 0, "A", "B")
        self.high_dom = wg(5, False, False, 0, "asld", 3.00, 3, 0, "A", "B")
        self.other_weight = wg(5, False, False, 0, "asld", 3.00, 5, 0, "A", "B")
        self.mut_weight = wg(1, True, False, 1.0, "asld", 3.00, 5, 0, "A", "B")
        self.mut_rate = wg(1, True, False, .99, "asld", 3.00, 5, 0, "A", "B")
        self.mut_dom = wg(1, True, False, .35, "asld", 3.00, 5, 0, "A", "B")
        dgeann.random.seed("vigor")

    def test_canread(self):
        result_a = self.unread_a.can_read(self.active_list, {})
        self.assertFalse(result_a)
        result_b = self.unread_b.can_read(self.active_list, {})
        self.assertFalse(result_b)
        result_true = self.low_dom_a.can_read(self.active_list, {})
        self.assertTrue(result_true)
        result = wg(1, False, False, 0, "A", 1.0, 0, 0, "Z", "Y").can_read(
                                                        {"X": 1, "Y": 1},
                                                        {"Z": "X"})
        self.assertTrue(result)
        result = wg(1, False, False, 0, "A", 1.0, 0, 0, "Z", "Y").can_read(
                                                        {"X": 1, "Y": 1}, {})
        self.assertFalse(result)

    def test_read(self):          
        #case where neither node can be read (one nonexistant input, one weight)
        test_a = self.unread_a.read(self.active_list, {}, self.unread_b)
        self.assertEqual(test_a, None)
        #case where only *this* node can be read
        test_b = self.low_dom_a.read(self.active_list, {}, self.unread_a)
        self.assertEqual(test_b, ["A", 3, "B", 0, 0.00])
        #case where only *other* node can be read
        test_c = self.unread_a.read(self.active_list, {}, self.low_dom_a)
        self.assertEqual(test_c, ["A", 3, "B", 0, 0.00])
        #case where this node dominates
        test_d = self.high_dom.read(self.active_list, {}, self.low_dom_a)
        self.assertEqual(test_d, ["A", 3, "B", 0, 3.00])
        #case where that node dominates
        test_e = self.low_dom_a.read(self.active_list, {}, self.high_dom)
        self.assertEqual(test_e, ["A", 3, "B", 0, 3.00])
        #case where they co-dominate
        test_f = self.low_dom_a.read(self.active_list, {}, self.low_dom_b)
        self.assertEqual(test_f, ["A", 3, "B", 0, 2.50])
        #case where they are both read, for different weights
        test_g = self.low_dom_a.read(self.active_list, {}, self.other_weight)
        self.assertEqual(test_g, ["A", 3, "B", 0, 0.00,
                                  "A", 5, "B", 0, 3.00])
        #case where there is no other gene
        test_h = self.low_dom_a.read(self.active_list, {})
        self.assertEqual(test_h, ["A", 3, "B", 0, 0.00])
        #sub_dict case
        test_i = wg(1, False, False, 0, "A", 1.0, 0, 0,
                                    "Z", "Y").read({"X": 1, "Y": 1}, {"Z": "X"})
        self.assertEqual(test_i, ["X", 0, "Y", 0, 1.0])

    def test_mutate(self):
        #case where mutatability is off
        test_mut_off = self.unread_a.mutate()
        self.assertEqual(test_mut_off, "")
        #case where mutability is on, dup off, but roll is higher than rate
        test_no_mut = self.mut_dom.mutate()
        self.assertEqual(test_no_mut, "")
        #case where mut is on, roll lower than rate (dom changes)
        dgeann.random.seed("vigor")
        for i in range(3):
            dgeann.random.random()
        test_dom_mut = self.mut_dom.mutate()
        self.assertEqual(test_dom_mut, "Dom, 1")
        #case where weight changes
        test_weight_mut = self.mut_weight.mutate()
        self.assertEqual(test_weight_mut, "Weight, -0.05954510163836234")
        #case where rate changes
        dgeann.random.seed("vigor")
        for i in range(7):
            dgeann.random.random()
        test_rate_mut = self.mut_rate.mutate()
        self.assertEqual(test_rate_mut, "Rate, -0.000574045945432189")
        #make sure clamping occurs
        self.mut_dom.dom = 5
        self.mut_dom.mut_rate = 1.0
        dgeann.random.seed("vigor")
        dgeann.weight_mut_probs = (0, 1, 0)
        for i in range(0):
            dgeann.random.random()
        test_clamp_mut = self.mut_dom.mutate()
        self.assertEqual(test_clamp_mut, "Dom, -1")
        self.mut_rate.mut_rate = 1
        dgeann.weight_mut_probs = (0, 0, 1)
        test_clamp_rate = self.mut_rate.mutate()
        self.assertEqual(test_clamp_rate, "Rate, -0.0004181538857107627")

#tests that layer genes read and mutate properly
class testLayer(unittest.TestCase):

    def setUp(self):
        self.unread_a = lg(5, False, False, 0, "askl", ["stairs"], 3, "IP")
        
    def test_makestring(self):
        dgeann.random.seed("genetic")
        data = lg(5, False, False, 0, "data", [], 9, "input")
        test_data = dedent("""\
                            input: \"data\"
                            input_shape: {
                              dim: 1
                              dim: 9
                            }
                            """)
        self.assertEqual(test_data, data.read_out({}, {}))
        action = lg(5, False, False, 0, "action", ["data", "STM"], 6, "IP")
        test_action = dedent('''\
                    layer {
                      name: "CEKMKT"
                      type: "Concat"

                      bottom: "data"
                      bottom: "STM"
                      top: "CEKMKT"
                      concat_param {
                        axis: 1
                      }
                    }
                    layer {
                      name: "action"
                      type: "InnerProduct"
                      param { lr_mult: 1 decay_mult: 1}
                      param { lr_mult: 2 decay_mult: 0}
                      inner_product_param {
                        num_output: 6
                        weight_filler {
                          type: "xavier"
                        }
                        bias_filler {
                          type: "constant"
                          value: 0
                        }
                      }
                      bottom: "CEKMKT"
                      top: "action"
                    }
                        ''')
        self.assertEqual(test_action, action.read_out({}, {"data": 9,
                                                            "STM": 5}))
        
    def test_read(self):
        active_list = {"cake": 5, "onion": 7}
        unread_b = lg(5, False, False, 0, "askl", ["cake", "onion", "strata"],
                      3, "IP")
        low_dom_a = lg(1, False, False, 0, "abcd", ["cake"], 4, "IP")
        low_dom_b = lg(1, False, False, 0, "efgh", ["cake"], 1, "IP")
        high_dom = lg(5, False, False, 0, "ijkl", ["cake"], 2, "IP")
        other_layer = lg(3, False, False, 0, "mnop", ["onion"], 3, "IP")
        multiple_layer = lg(3, False, False, 0, "qrst", ["cake", "onion"],
                            3, "IP")
        no_inputs = lg(3, False, False, 0, "uvwx", [], 3, "input")
        stm_input = lg(5, False, False, 0, "stm_input", ["data", "reward"],
                       6, "input")
        stm = lg(5, False, False, 0, "STM", ["stm_input"], 6, "STMlayer")
        dgeann.random.seed("layers")
        concat_dict = {}
        #a case where neither is read
        test_a = self.unread_a.read(active_list, unread_b, {}, {})
        self.assertEqual(test_a, None)
        #a case where the first is read
        test_b = low_dom_a.read(active_list, self.unread_a, {}, {})
        self.assertEqual(test_b, low_dom_a)
        #a case where the second is read
        test_c = self.unread_a.read(active_list, low_dom_a, {}, {})
        self.assertEqual(test_c, low_dom_a)
        #a case where they have the same dom
        test_d = low_dom_a.read(active_list, low_dom_b, {}, {})
        self.assertEqual(test_d, low_dom_b)
        #a case where the first dominates
        test_e = high_dom.read(active_list, low_dom_a, {}, {})
        self.assertEqual(test_e, high_dom)
        #a case where the second dominates
        test_f = low_dom_b.read(active_list, high_dom, {}, {})
        self.assertEqual(test_f, high_dom)
        #a case with null other gene (and two dependencies)
        null = lg(0, False, False, 0, "null", [], None, None)
        test_g = multiple_layer.read(active_list, null, {}, {})
        self.assertEqual(test_g, multiple_layer)
        #a case with no other gene (and no dependencies)
        test_h = no_inputs.read(active_list, null, {}, {})
        self.assertEqual(test_h, no_inputs)
        #a case where the other gene has no dependencies
        test_i = low_dom_a.read(active_list, no_inputs, {}, {})
        self.assertEqual(test_i, no_inputs)
        #a case where other gene is already in active_list
        active_list["abcd"] = 4
        test_j = low_dom_b.read(active_list, low_dom_a, {}, {})
        self.assertEqual(test_j, low_dom_b)
        
    def test_mutate(self):
        mut_rate = lg(3, True, False, .9, "askl", [], 3, "input")
        mut_dom = lg(3, True, False, 1.0, "askl2", [], 3, "input")
        mut_num = lg(3, True, False, 1.0, "askl", [], 5, "input")
        mut_dup = lg(3, True, True, 1.0, "askl", [], 5, "input")
        mut_add_input = lg(3, True, False, 1.0, "askl", [], 5, "")
        dgeann.random.seed("vigor")
        test_mut_off = self.unread_a.mutate()
        self.assertEqual(test_mut_off, "")
        dgeann.random.random()
        dgeann.random.random()
        test_mut_dom = mut_dom.mutate()
        self.assertEqual(test_mut_dom, "Dom, 1")
        dgeann.random.seed("vigor")
        test_mut_rate = mut_rate.mutate()
        self.assertEqual(test_mut_rate, "Rate, 1.0064147264152894e-05")
        dgeann.random.random()
        dgeann.random.random()
        test_mut_num = mut_num.mutate()
        self.assertEqual(test_mut_num, "Nodes, 1")
        dgeann.random.random()
        dgeann.random.random()
        dgeann.random.random()
        test_mut_dup = mut_dup.mutate()
        self.assertEqual(test_mut_dup, "Duplicate,")
        dgeann.random.seed("vigor")
        for i in range(7):
            dgeann.random.random()
        test_mut_input = mut_add_input.mutate()
        self.assertEqual(test_mut_input, "Add input,")
        mut_dom.mut_rate = 0.01
        test_mut_none = mut_dom.mutate()
        self.assertEqual(test_mut_none, "")
        mut_rate.mut_rate = 1
        #making sure that results can't be out of bounds
        dgeann.random.seed("vigor")
        test_mut_clampedrate = mut_rate.mutate()
        self.assertTrue(float(test_mut_clampedrate[6:]) < 0)
        dgeann.random.seed("evodevo")
        dgeann.random.random()
        dgeann.random.random()
        mut_num.nodes = 1
        test_mut_clampednum = mut_num.mutate()
        self.assertEqual(test_mut_clampednum, "Nodes, 1")
        mut_dom.dom = 5
        mut_dom.mut_rate = 1.0
        dgeann.random.seed("vigor")
        dgeann.random.random()
        dgeann.random.random()
        test_mut_clampeddom = mut_dom.mutate()
        self.assertEqual(test_mut_clampeddom, "Dom, -1")
        

#tests that genome builds a new creature properly
#along with tests for helper functions
class testBuild(unittest.TestCase):

    def setUp(self):
        self.active_list = {"data": 8, "concat_0": 5}
        unread_a = wg(1, False, False, 0, "asld", 0.00, 0, 0, "N", "n")
        unread_b = wg(1, False, False, 0, "asld",  0.00, 0, 0, "A", "B")
        low_dom_a = wg(1, False, False, 0, "abcd",  0.00, 0, 0,
                       "data", "action")
        low_dom_b = wg(1, False, False, 0, "lowdb",  5.00, 0, 0,
                       "data", "action")
        low_dom_c = wg(1, False, False, 0, "jklm", 7.00, 0, 1, "data", "action")
        low_dom_d = wg(1, False, False, 0, "lowdd", 1.00, 4, 0,
                       "data", "action")
        high_dom = wg(5, False, False, 0, "xyz", 3.00, 4, 0, "data", "action")
        other_weight = wg(5, False, False, 0, "other", 3.00, 4, 4,
                          "data", "action")
        #case where neither read, both read (same gene)
        #a is read first, low_dom_d and high_dom both read
        #other_weight is left to read on b
        alist = [unread_a, low_dom_a, low_dom_c, high_dom]
        blist = [unread_b, low_dom_b, low_dom_d, other_weight]

        self.l_unread_a = lg(5, False, False, 0, "askl", ["stairs"], 3, "IP")
        l_unread_b = lg(5, False, False, 0, "askl",
                        ["cake", "onion", "strata"], 3, "IP")
        self.l_low_dom_a = lg(1, False, False, 0, "data", [], 8, "input")
        self.l_low_dom_b = lg(1, False, False, 0, "efgh", [], 5, "input")
        self.l_low_dom_c = lg(1, False, False, 0, "qrst", [], 4, "input")
        l_low_dom_d = lg(1, False, False, 0, "uvwx", [], 5, "input")
        self.l_high_dom = lg(5, False, False, 0, "ijkl", [], 2, "input")
        self.l_dummy_layer = lg(5, False, False, 0, "blegh", [], 5, "input")
        l_action_layer = lg(3, False, False, 0, "action",
                            ["data", "blegh"], 5, "IP")
        #case where neither read, both readable(rng set to read a)
        #a can be read, b can be read,
        #a has dom, a is longer by three genes
        clist = [self.l_unread_a, self.l_low_dom_a, self.l_low_dom_c, l_unread_b,
                 self.l_high_dom, self.l_dummy_layer, l_action_layer]
        dlist = [l_unread_b, self.l_low_dom_b, self.l_unread_a,
                 l_low_dom_d, self.l_low_dom_b]
        
        self.test_genome_a = dgeann.Genome(clist, dlist, alist, blist)
        self.data = lg(5, False, False, 0, "data", [], 9, "input")
        self.reward = lg(5, False, False, 0, "reward", [], 6, "input")
        self.stm_input = lg(5, False, False, 0, "stm_input",
                            ["data", "reward"], 6, "input")
        self.stm = lg(5, False, False, 0, "STM", ["stm_input"], 6, "STMlayer")
        self.concat = lg(5, False, False, 0, "concat_0",
                         ["data", "STM"], None, "concat")
        self.action = lg(5, False, False, 0, "action", ["concat_0"], 6, "IP")
        self.loss = lg(5, False, False, 0, "loss",
                       ["action", "reward"], 6, "loss")

    def test_net_ident(self):
        dgeann.random.seed("genetic")
        ident = dgeann.network_ident()
        self.assertEqual(ident, "T125-659-499")

    def test_layers_equalize_simple(self):
        self.l_unread_a.inputs = []
        self.l_unread_a.layer_type = 'input'
        #layers_equalize
        null = lg(3, False, False, 0, "null", [], None, None)
        #case where both genomes are the same length
        g1 = dgeann.Genome([self.l_unread_a, self.l_low_dom_a,
                            self.l_dummy_layer],
                           [self.l_unread_a, self.l_high_dom,
                            self.l_dummy_layer], [], [])
        g1.layers_equalize()
        g1_test = dgeann.Genome([self.l_unread_a, self.l_low_dom_a,
                            self.l_dummy_layer],
                           [self.l_unread_a, self.l_high_dom,
                            self.l_dummy_layer], [], [])
        self.assertEqual(len(g1.layerchr_a), len(g1_test.layerchr_a))
        self.assertEqual(len(g1.layerchr_b), len(g1_test.layerchr_b))
        for i in range(len(g1.layerchr_a)):
            self.assertEqual(g1.layerchr_a[i].ident,
                             g1_test.layerchr_a[i].ident)
        for i in range(len(g1.layerchr_b)):
            self.assertEqual(g1.layerchr_b[i].ident,
                             g1_test.layerchr_b[i].ident)
            
    def test_layers_equalize_unequal(self):
        #case where chr a has an extra gene
        null = lg(3, False, False, 0, "null", [], None, None)
        g2 = dgeann.Genome([self.l_unread_a, self.l_low_dom_a,
                            self.l_low_dom_b, self.l_dummy_layer],
                           [self.l_unread_a, self.l_high_dom,
                            self.l_dummy_layer], [], [])
        g2.layers_equalize()
        g2_test = dgeann.Genome([self.l_unread_a, self.l_low_dom_a,
                            self.l_low_dom_b, self.l_dummy_layer],
                           [self.l_unread_a, self.l_high_dom, null,
                            self.l_dummy_layer], [], [])
        self.assertEqual(len(g2.layerchr_a), len(g2_test.layerchr_a))
        self.assertEqual(len(g2.layerchr_b), len(g2_test.layerchr_b))
        for i in range(len(g2.layerchr_a)):
            self.assertEqual(g2.layerchr_a[i].ident,
                             g2_test.layerchr_a[i].ident)
        for i in range(len(g2.layerchr_b)):
            self.assertEqual(g2.layerchr_b[i].ident,
                             g2_test.layerchr_b[i].ident)
        #case where chr b has an extra gene
        g3 = dgeann.Genome([self.l_unread_a, self.l_low_dom_a,
                            self.l_dummy_layer],
                           [self.l_unread_a, self.l_low_dom_b, self.l_high_dom,
                            self.l_dummy_layer], [], [])
        g3.layers_equalize()
        g3_test = dgeann.Genome([self.l_unread_a, self.l_low_dom_a, null,
                            self.l_dummy_layer],
                           [self.l_unread_a, self.l_low_dom_b, self.l_high_dom,
                            self.l_dummy_layer], [], [])
        self.assertEqual(len(g3.layerchr_a), len(g3_test.layerchr_a))
        self.assertEqual(len(g3.layerchr_b), len(g3_test.layerchr_b))
        for i in range(len(g3.layerchr_a)):
            self.assertEqual(g3.layerchr_a[i].ident,
                             g3_test.layerchr_a[i].ident)
        for i in range(len(g3.layerchr_b)):
            self.assertEqual(g3.layerchr_b[i].ident,
                             g3_test.layerchr_b[i].ident)
        #case where chr a has two extra genes
        g4 = dgeann.Genome([self.l_unread_a, self.l_low_dom_a, self.l_low_dom_b,
                            self.l_low_dom_c, self.l_dummy_layer],
                            [self.l_unread_a, self.l_low_dom_b, self.l_high_dom,
                            self.l_dummy_layer], [], [])
        g4.layers_equalize()
        g4_test = dgeann.Genome([self.l_unread_a, self.l_low_dom_a,
                                 self.l_low_dom_b, self.l_low_dom_c,
                                 self.l_dummy_layer],
                            [self.l_unread_a, self.l_low_dom_b,
                             self.l_high_dom, null, self.l_dummy_layer], [], [])
        self.assertEqual(len(g4.layerchr_a), len(g4_test.layerchr_a))
        self.assertEqual(len(g4.layerchr_b), len(g4_test.layerchr_b))
        for i in range(len(g4.layerchr_a)):
            self.assertEqual(g4.layerchr_a[i].ident,
                             g4_test.layerchr_a[i].ident)
        for i in range(len(g4.layerchr_b)):
            self.assertEqual(g4.layerchr_b[i].ident,
                             g4_test.layerchr_b[i].ident)
        #case where chr b has three extra genes
        g5 = dgeann.Genome([self.l_unread_a, self.l_low_dom_a,
                            self.l_dummy_layer],
                           [self.l_unread_a, self.l_low_dom_c,
                            self.l_low_dom_b, self.l_low_dom_a,
                            self.l_high_dom, self.l_dummy_layer], [], [])
        g5.layers_equalize()
        g5_test = dgeann.Genome([self.l_unread_a, null, null, self.l_low_dom_a,
                                 null, self.l_dummy_layer],
                           [self.l_unread_a, self.l_low_dom_c,
                            self.l_low_dom_b, self.l_low_dom_a,
                            self.l_high_dom, self.l_dummy_layer], [], [])
        self.assertEqual(len(g5.layerchr_a), len(g5_test.layerchr_a))
        self.assertEqual(len(g5.layerchr_b), len(g5_test.layerchr_b))
        for i in range(len(g5.layerchr_a)):
            self.assertEqual(g5.layerchr_a[i].ident,
                             g5_test.layerchr_a[i].ident)
        for i in range(len(g5.layerchr_b)):
            self.assertEqual(g5.layerchr_b[i].ident,
                             g5_test.layerchr_b[i].ident)
            
    def test_layers_equalize_IPlayers(self):
        null = lg(3, False, False, 0, "null", [], None, None)
        #case where chromosomes are equal, but with IP layers
        IPG = lg(5, False, False, 0, "G", ["askl"], 2, "IP")
        IPH = lg(5, False, False, 0, "H", ["askl"], 2, "IP")
        IPI = lg(5, False, False, 0, "I", ["G"], 2, "IP")
        IPJ = lg(5, False, False, 0, "J", ["H"], 2, "IP")
        IPK = lg(5, False, False, 0, "I", ["G", "H"], 2, "IP")
        g6 = dgeann.Genome([self.l_unread_a, IPG, IPI],
                            [self.l_unread_a, IPH, IPJ], [], [])
        g6.layers_equalize()
        self.assertEqual(len(g6.layerchr_a), 3)
        self.assertEqual(len(g6.layerchr_b), 3)
        #case where chromosomes are unequal, with an IP layer with high dom
        g7 = dgeann.Genome([self.l_unread_a, IPG, IPH, IPK],
                            [self.l_unread_a, IPG, IPI], [], [])
        g7.layers_equalize()
        g7_test_b = [self.l_unread_a, IPG, null, IPI]
        self.assertEqual(len(g7.layerchr_a), 4)
        self.assertEqual(len(g7.layerchr_b), 4)
        for i in range(4):
            self.assertEqual(g7.layerchr_b[i].ident, g7_test_b[i].ident)
            
    def test_structure_network(self):
        null = lg(3, False, False, 0, "null", [], None, None)
        g1 = dgeann.Genome([self.l_unread_a, self.l_low_dom_a,
                            self.l_dummy_layer],
                           [self.l_unread_a, self.l_high_dom,
                            self.l_dummy_layer], [], [])
        g2 = dgeann.Genome([self.l_unread_a, self.l_low_dom_a,
                            self.l_low_dom_b, self.l_dummy_layer],
                           [self.l_unread_a, self.l_high_dom,
                            self.l_dummy_layer], [], [])
        g2.layers_equalize()
        g3 = dgeann.Genome([self.l_unread_a, self.l_low_dom_a,
                            self.l_dummy_layer],
                           [self.l_unread_a, self.l_low_dom_b, self.l_high_dom,
                            self.l_dummy_layer], [], [])
        g3.layers_equalize()
        g4 = dgeann.Genome([self.l_unread_a, self.l_low_dom_a, self.l_low_dom_b,
                            self.l_low_dom_c, self.l_dummy_layer],
                            [self.l_unread_a, self.l_low_dom_b, self.l_high_dom,
                            self.l_dummy_layer], [], [])
        g4.layers_equalize()
        g5 = dgeann.Genome([self.l_unread_a, self.l_low_dom_a,
                            self.l_dummy_layer],
                           [self.l_unread_a, self.l_low_dom_c,
                            self.l_low_dom_b, self.l_low_dom_a,
                            self.l_high_dom, self.l_dummy_layer], [], [])
        g5.layers_equalize()
        IPG = lg(5, False, False, 0, "G", ["askl"], 2, "IP")
        IPH = lg(5, False, False, 0, "H", ["askl"], 2, "IP")
        IPI = lg(5, False, False, 0, "I", ["G"], 2, "IP")
        IPJ = lg(5, False, False, 0, "J", ["H"], 2, "IP")
        IPK = lg(5, False, False, 0, "K", ["G", "H"], 2, "IP")
        askl = lg(1, False, False, 0, "askl", [], 2, "data")
        g6 = dgeann.Genome([askl, IPG, IPI],
                            [askl, IPH, IPJ], [], [])
        g7 = dgeann.Genome([askl, IPG, IPH, IPK],
                            [askl, IPG, IPI], [], [])
        g7.layers_equalize()
        dgeann.random.seed("evo-devo")
        g1_sub, g1_list, g1_layout = g1.structure_network({})
        g2_sub, g2_list, g2_layout = g2.structure_network({})
        g3_sub, g3_list, g3_layout = g3.structure_network({})
        g4_sub, g4_list, g4_layout = g4.structure_network({})
        g5_sub, g5_list, g5_layout = g5.structure_network({})
        g6_sub, g6_list, g6_layout = g6.structure_network({})
        g7_sub, g7_list, g7_layout = g7.structure_network({})
        g1_testlist = {"ijkl": 2, "blegh": 5}
        g2_testlist = {"ijkl": 2, "blegh": 5}
        g3_testlist = {"data": 8, "blegh": 5, "ijkl": 2}
        g4_testlist = {"efgh": 5, "blegh": 5, "ijkl": 2}
        g5_testlist = {"data": 8, "ijkl": 2, "blegh": 5}
        g6_testlist = {"askl": 2, "G": 2, "J": 2}
        #substitution happens here
        g7_testlist = {"askl": 2, "G": 2, "I": 2, "K": 2}
        g1_testlayout = [self.l_high_dom, self.l_dummy_layer]
        g2_testlayout = [self.l_high_dom, self.l_dummy_layer]
        g3_testlayout = [self.l_low_dom_a, self.l_high_dom, self.l_dummy_layer]
        g4_testlayout = [self.l_low_dom_b, self.l_high_dom, self.l_dummy_layer]
        g5_testlayout = [self.l_low_dom_a, self.l_high_dom, self.l_dummy_layer]
        g6_testlayout = [askl, IPG, IPJ]
        g7_testlayout = [askl, IPG, IPI, IPK]
        listlists = [g1_list, g2_list, g3_list, g4_list, g5_list, g6_list,
                     g7_list]
        testlists = [g1_testlist, g2_testlist, g3_testlist, g4_testlist,
                     g5_testlist, g6_testlist, g7_testlist]
        listlayouts = [g1_layout, g2_layout, g3_layout, g4_layout,
                     g5_layout, g6_layout, g7_layout]
        testlayouts = [g1_testlayout, g2_testlayout, g3_testlayout, g4_testlayout,
                     g5_testlayout, g6_testlayout, g7_testlayout]
        for i in range(len(listlists)):
            self.assertEqual(listlists[i], testlists[i])
            j = 0
            for gene in listlayouts[i]:
                self.assertEqual(gene.ident, testlayouts[i][j].ident)
                j += 1
        self.assertEqual(g6.layerchr_a[2].inputs, ["G"])
        self.assertEqual(g6_layout[2].inputs, ["G"])

    def test_build_layers_text(self):
        askl = lg(1, False, False, 0, "askl", [], 3, "input")
        IPG = lg(5, False, False, 0, "G", ["askl"], 2, "IP")
        IPH = lg(5, False, False, 0, "H", ["askl"], 2, "IP")
        IPI = lg(5, False, False, 0, "I", ["G"], 2, "IP")
        IPJ = lg(5, False, False, 0, "J", ["H"], 2, "IP")
        IPK = lg(5, False, False, 0, "I", ["G", "H"], 2, "IP")
        g6 = dgeann.Genome([askl, IPG, IPI],
                            [askl, IPH, IPJ], [], [])
        g7 = dgeann.Genome([askl, IPG, IPH, IPK],
                            [askl, IPG, IPI], [], [])
        g7.layers_equalize()
        dgeann.random.seed("evo-devo")
        g6_list, g6_concats, g6_sub = g6.build_layers({}, "g6.txt", {})
        g6_test = dedent('''\
                        input: "askl"
                        input_shape: {
                          dim: 1
                          dim: 3
                        }
                        layer {
                          name: "G"
                          type: "InnerProduct"
                          param { lr_mult: 1 decay_mult: 1}
                          param { lr_mult: 2 decay_mult: 0}
                          inner_product_param {
                            num_output: 2
                            weight_filler {
                              type: "xavier"
                            }
                            bias_filler {
                              type: "constant"
                              value: 0
                            }
                          }
                          bottom: "askl"
                          top: "G"
                        }
                        layer {
                          name: "I"
                          type: "InnerProduct"
                          param { lr_mult: 1 decay_mult: 1}
                          param { lr_mult: 2 decay_mult: 0}
                          inner_product_param {
                            num_output: 2
                            weight_filler {
                              type: "xavier"
                            }
                            bias_filler {
                              type: "constant"
                              value: 0
                            }
                          }
                          bottom: "G"
                          top: "I"
                        }
                        ''')
        
        f = open("g6_test.txt", "a")
        f.write(g6_test)
        f.close()
        with open("g6.txt") as file:
            with open("g6_test.txt") as file2:
                for line, line2 in zip(file, file2):
                    self.assertEqual(line, line2)
        g7_list, g7_concats, g7_sub = g7.build_layers({}, "g7.txt", {})
        g7_test = dedent('''\
                        input: "askl"
                        input_shape: {
                          dim: 1
                          dim: 3
                        }
                        layer {
                          name: "G"
                          type: "InnerProduct"
                          param { lr_mult: 1 decay_mult: 1}
                          param { lr_mult: 2 decay_mult: 0}
                          inner_product_param {
                            num_output: 2
                            weight_filler {
                              type: "xavier"
                            }
                            bias_filler {
                              type: "constant"
                              value: 0
                            }
                          }
                          bottom: "askl"
                          top: "G"
                        }
                        layer {
                          name: "H"
                          type: "InnerProduct"
                          param { lr_mult: 1 decay_mult: 1}
                          param { lr_mult: 2 decay_mult: 0}
                          inner_product_param {
                            num_output: 2
                            weight_filler {
                              type: "xavier"
                            }
                            bias_filler {
                              type: "constant"
                              value: 0
                            }
                          }
                          bottom: "askl"
                          top: "H"
                        }
                        layer {
                          name: "OEHJCT"
                          type: "Concat"

                          bottom: "G"
                          bottom: "H"
                          top: "OEHJCT"
                          concat_param {
                            axis: 1
                          }
                        }
                        layer {
                          name: "I"
                          type: "InnerProduct"
                          param { lr_mult: 1 decay_mult: 1}
                          param { lr_mult: 2 decay_mult: 0}
                          inner_product_param {
                            num_output: 2
                            weight_filler {
                              type: "xavier"
                            }
                            bias_filler {
                              type: "constant"
                              value: 0
                            }
                          }
                          bottom: "OEHJCT"
                          top: "I"
                        }
                        ''')
        f = open("g7_test.txt", "a")
        f.write(g7_test)
        f.close()
        with open("g7.txt") as file:
            with open("g7_test.txt") as file2:
                for line, line2 in zip(file, file2):
                    self.assertEqual(line, line2)
        os.remove("g6.txt")
        os.remove("g7.txt")
        os.remove("g6_test.txt")
        os.remove("g7_test.txt")
        
    def test_build_layers_null(self):
        #case based on chopped-down real example
        #should compare null against can't-be-read ("P" when "M" is not read)
        dgeann.random.seed("evo|devo")
        d = lg(5, False, False, 0, "d", [], 1, "input")
        V = lg(3, False, False, 0, "V", ["d"], 1, "IP")
        M = lg(3, False, False, 0, "M", ["d"], 1, "IP")
        P = lg(3, False, False, 0, "P", ["d", "M"], 1, "IP")
        ip1 = lg(3, False, False, 0, "IP", ["d", "M", "P"], 1, "IP")
        out1 = lg(3, False, False, 0, "out", ["IP", "V"], 1, "IP")
        ip2 = lg(3, False, False, 0, "IP", ["d"], 1, "IP")
        out2 = lg(3, False, False, 0, "out", ["IP"], 1, "IP")
        g8 = dgeann.Genome([d, V, M, P, ip1, out1], [d, ip2, out2], [], [],
                           ["out"])
        g8_list, g8_concats, g8_sub = g8.build_layers({}, "g8.txt", {})
        self.assertEqual(g8_list, {"d": 1, "V": 1, "IP": 1, "out": 1})
        os.remove('g8.txt')

    def test_adjust_weights(self):
        testa = wg(1, False, False, 0.0, "a", -0.06807647, 0, 0,
                   "data", "action")
        testb = wg(1, False, False, 0.0, "a", -0.08293831, 0, 1,
                   "data", "action")
        testc = wg(1, False, False, 0.0, "a", -0.3910988, 0, 2,
                   "data", "action")
        act = lg(5, False, False, 0, "action", ["data"], 3, "IP")
        gen = dgeann.Genome([self.data, act], [self.data, act],
                            [testa, testb, testc], [testa, testb, testc])
        solv = gen.build()   
        self.assertNotEqual(solv.net.params["action"][0].data[0][0],
                            -0.06807647)
        val0 = ["data", 0, "action", 0, -0.06807647]
        dgeann.Genome.adjust_weight(solv.net, val0)
        self.assertAlmostEqual(solv.net.params["action"][0].data[0][0],
                               -0.06807647)
        val1 = ["data", 1, "action", 0, -0.08293831,
                "data", 2, "action", 0, -0.3910988]
        self.assertNotEqual(solv.net.params["action"][0].data[0][1],
                            -0.08293831)
        self.assertNotEqual(solv.net.params["action"][0].data[0][2],
                            -0.3910988)
        dgeann.Genome.adjust_weight(solv.net, val1)
        self.assertAlmostEqual(solv.net.params["action"][0].data[0][1],
                               -0.08293831)
        self.assertAlmostEqual(solv.net.params["action"][0].data[0][2],
                              -0.3910988)

    def test_read_through(self):
        act = lg(5, False, False, 0, "action", ["data"], 5, "IP")
        gen = dgeann.Genome([self.data, act], [self.data, act], [], [])
        solv = gen.build()
        net = solv.net
        #case where gene is not read
        n, a = self.test_genome_a.read_through("a", 0, [], net, [])
        self.assertEqual(n, 1)
        self.assertEqual(a.ident, "abcd")
        #case where gene is read
        self.assertNotEqual(net.params["action"][0].data[0][3], 0.00)
        active_list = {"data": 8, "action": 5}
        n, a = self.test_genome_a.read_through("a", 1, active_list, net, [])
        self.assertAlmostEqual(net.params["action"][0].data[0][0], 0.00)
        self.assertEqual(n, 2)
        self.assertEqual(a.ident, "jklm")
        #case where gene is last on chromosome
        self.assertNotEqual(net.params["action"][0].data[0][4], 3.00)
        n, a = self.test_genome_a.read_through("a", 3, active_list, net, [])
        self.assertAlmostEqual(net.params["action"][0].data[0][4], 3.00)
        self.assertEqual(n, 4)
        self.assertEqual(a.ident, "xyz")

    def test_concats(self):
        a_u_00 = wg(5, False, False, 0, "au00",  3.00, 0, 0, "INa", "IPu")
        a_o_00 = wg(5, False, False, 0, "ao00", 3.00, 0, 0, "INa", "IPo")
        i_u_00 = wg(5, False, False, 0, "iu00", 5.00, 0, 0, "INi", "IPu")
        i_o_00 = wg(5, False, False, 0, "io00", 5.00, 0, 0, "INi", "IPo")
        weight_list = [a_u_00, a_o_00, i_u_00, i_o_00]
        concat_genome = dgeann.Genome([], [], weight_list, weight_list)
        active_list = {"INa": 5, "INi": 5, "concat": 10, "IPu": 5, "IPo": 5}
        concat_dict = {"concat": [["INa", "INi"], [5, 5], ["IPu", "IPo"]]}
        concat_genome.concat_adjust(concat_dict)
        cwa = concat_genome.weightchr_a
        cwb = concat_genome.weightchr_b
        results = [0, 0, 5, 5]
        for i in range(len(results)):
            self.assertEqual(cwa[i].alt_in, results[i])
            self.assertEqual(cwb[i].alt_in, results[i])
        #test with two inputs
        e_u_00 = wg(5, False, False, 0, "ie00", 3.00, 0, 0, "INe", "IPu")
        e_o_00 = wg(5, False, False, 0, "ie00", 3.00, 0, 0, "INe", "IPo")
        weight_list2 = [a_u_00, a_o_00, i_u_00, i_o_00, e_u_00, e_o_00]
        for weight in weight_list2:
            weight.alt_in = weight.in_node
        concat_genome2 = dgeann.Genome([], [], weight_list2, weight_list2)
        active_list2 = {"INa": 5, "INi": 5, "INe": 5, "concat": 10, "IPu": 5,
                        "IPo": 5}
        concat_dict2 = {"concat": [["INa", "INi", "INe"], [5, 5],
                                   ["IPu", "IPo"]]}
        concat_genome2.concat_adjust(concat_dict2)
        cwa = concat_genome2.weightchr_a
        cwb = concat_genome2.weightchr_b
        for i in range(len(results)):
            self.assertEqual(cwa[i].alt_in, results[i])
            self.assertEqual(cwb[i].alt_in, results[i])
        
    def test_build_weights(self):
        act = lg(5, False, False, 0, "action", ["data"], 5, "IP")
        gen = dgeann.Genome([self.data, act], [self.data, act], [], [])
        solv = gen.build()
        net = solv.net
        data = net.params["action"][0].data
        active_list = {"data": 8, "action": 5}
        #simple test
        zz_genea = wg(1, False, False, 0, "zzga", 1.00, 0, 0, "data", "action")
        zz_geneb = wg(1, False, False, 0, "zzgb", 5.00, 0, 0, "data", "action")
        zo_genea = wg(1, False, False, 0, "abcd", 1.00, 0, 1, "data", "action")
        zo_geneb = wg(1, False, False, 0, "abcd", 5.00, 0, 1, "data", "action")
        genome_a = dgeann.Genome([], [], [zz_genea, zo_genea], [zz_geneb,
                                                                zo_geneb])
        genome_a.build_weights(active_list, net, {})
        self.assertAlmostEqual(data[0][0], 3.00)
        self.assertAlmostEqual(data[1][0], 3.00)
        #test where a is one longer
        zt_genea = wg(1, False, False, 0, "mnop", 1.00, 0, 2, "data", "action")
        genome_b = dgeann.Genome([], [], [zz_genea, zo_genea, zt_genea],
                    [zz_geneb, zo_geneb])
        genome_b.build_weights(active_list, net, {})
        self.assertAlmostEqual(data[2][0], 1.00)
        #test where b is one longer
        zt_geneb = wg(1, False, False, 0, "mnop", 5.00, 0, 2, "data", "action")
        genome_c = dgeann.Genome([], [], [zz_genea, zo_genea],
                    [zz_geneb, zo_geneb, zt_geneb])
        genome_c.build_weights(active_list, net, {})
        self.assertAlmostEqual(data[2][0], 5.00)
        #test where a pulls ahead
        oz_genea = wg(1, False, False, 0, "efgh", 1.00, 1, 0, "data", "action")
        genome_d = dgeann.Genome([], [], [zz_genea, oz_genea],
                                   [zz_geneb, zo_geneb])
        genome_d.build_weights(active_list, net, {})
        self.assertAlmostEqual(data[0][1], 1.00)
        #test where b pulls ahead
        oz_geneb = wg(1, False, False, 0, "ijkl", 5.00, 1, 0, "data", "action")
        genome_e = dgeann.Genome([], [], [zz_genea, zo_genea],
                                   [zz_geneb, oz_geneb])
        genome_e.build_weights(active_list, net, {})
        self.assertAlmostEqual(data[0][1], 5.00)
        #test where b out gets bigger while a is at 0/0
        genome_f = dgeann.Genome([], [], [zo_genea, zz_genea, zt_genea],
                                   [zo_geneb, zt_geneb, zz_geneb])
        genome_f.build_weights(active_list, net, {})
        self.assertAlmostEqual(data[0][0], 3.00)
        data[0][0] = 7.00
        #test where a out gets bigger while b is at 0/0
        genome_g = dgeann.Genome([], [], [zo_genea, zt_geneb, zz_genea],
                                   [zo_geneb, zz_geneb, zt_geneb])
        genome_g.build_weights(active_list, net, {})
        self.assertAlmostEqual(data[0][0], 3.00)
        data[0][0] = 7.00
        #test where a goes to 0/0 while b is still going
        ff_genea = wg(1, False, False, 0, "ffga", 1.00, 5, 4, "data", "action")
        ff_geneb = wg(1, False, False, 0, "ffgb", 5.00, 5, 4, "data", "action")
        sz_geneb = wg(1, False, False, 0, "szgb", 5.00, 6, 0, "data", "action")
        so_geneb = wg(1, False, False, 0, "sogb", 5.00, 6, 1, "data", "action")
        genome_j = dgeann.Genome([], [], [ff_genea, zz_genea],
                                   [ff_geneb, sz_geneb, so_geneb, zz_geneb])
        genome_j.build_weights(active_list, net, {})
        self.assertAlmostEqual(data[0][0], 3.00)
        #test where b goes to 0/0 while a is still going
        data[0][0] = 7.00
        sz_genea = wg(1, False, False, 0, "szga", 1.00, 6, 0, "data", "action")
        so_genea = wg(1, False, False, 0, "szga", 1.00, 6, 1, "data", "action")
        genome_l = dgeann.Genome([], [], [ff_genea, sz_genea, so_genea,
                                          zz_genea],
                                         [ff_geneb, zz_geneb])
        genome_l.build_weights(active_list, net, {})
        self.assertAlmostEqual(data[0][0], 3.00)
   
    def test_build(self):
        dgeann.random.seed("genetics")
        solv = self.test_genome_a.build(delete=False)
        x = os.path.join('Gen files', 'T703-392-979.gen')
        os.remove(x)
        data = solv.net.params['action'][0].data
        self.assertAlmostEqual(data[0][0], 2.50)
        self.assertAlmostEqual(data[1][0], 7.00)
        self.assertAlmostEqual(data[0][4], 3.00)
        self.assertAlmostEqual(data[4][4], 3.00)
        #test concats
        layer_a = lg(5, False, False, 0, "INa", [], 5, "input")
        layer_i = lg(5, False, False, 0, "INi", [], 5, "input")
        layer_u = lg(5, False, False, 0, "IPu", ["INa", "INi"], 5, "IP")
        layer_o = lg(5, False, False, 0, "IPo", ["INa", "INi"], 5, "IP")
        layer_list = [layer_a, layer_i, layer_u, layer_o]
        a_u_00 = wg(5, False, False, 0, "au00", 3.00, 0, 0, "INa", "IPu")
        a_o_00 = wg(5, False, False, 0, "ao00", 3.00, 0, 0, "INa", "IPo")
        i_u_00 = wg(5, False, False, 0, "iu00", 5.00, 0, 0, "INi", "IPu")
        i_o_00 = wg(5, False, False, 0, "io00", 5.00, 0, 0, "INi", "IPo")
        weight_list = [a_u_00, a_o_00, i_u_00, i_o_00]
        concat_genome = dgeann.Genome(layer_list, layer_list, weight_list,
                                        weight_list)
        solv = concat_genome.build(delete=False)
        data_u = solv.net.params['IPu'][0].data
        data_o = solv.net.params['IPo'][0].data        
        x = os.path.join('Gen files', 'T176-249-355.gen')
        os.remove(x)
        self.assertAlmostEqual(data_u[0][0], 3.00)
        self.assertAlmostEqual(data_u[0][5], 5.00)
        self.assertAlmostEqual(data_o[0][0], 3.00)
        self.assertAlmostEqual(data_o[0][5], 5.00)

    def test_build_subs(self):
        #test that subbing layers works correctly
        test_subs = dgeann.Genome([lg(1, False, False, 0, "A", [], 2, "input"),
                                   lg(5, False, False, 0, "C", ["A"], 2, "IP")],
                                  [lg(5, False, False, 0, "B", [], 1, "input"),
                                   lg(1, False, False, 0, "D", ["B"], 2, "IP")],
                                  [wg(3, False, False, 0, "a", 3.0, 0, 0,
                                      "A", "C"),
                                   wg(3, False, False, 0, "b", 3.0, 0, 1,
                                      "A", "C"),
                                   wg(3, False, False, 0, "c", 3.0, 1, 0,
                                      "A", "C"),
                                   wg(3, False, False, 0, "d", 3.0, 1, 1,
                                      "A", "C")],
                                  [wg(3, False, False, 0, "e", 1.0, 0, 0,
                                      "B", "D"),
                                   wg(3, False, False, 0, "f", 1.0, 0, 1,
                                      "B", "D"),
                                   wg(3, False, False, 0, "g", 1.0, 1, 0,
                                      "B", "D"),
                                   wg(3, False, False, 0, "h", 1.0, 1, 1,
                                      "B", "D")])
        subs = test_subs.build()
        self.assertEqual(len(subs.net.params["C"][0].data[0]), 1)
        self.assertEqual(subs.net.params["C"][0].data[0][0], 3.0)
        self.assertEqual(subs.net.params["C"][0].data[1][0], 3.0)        
        

#tests the function to turn random network weights into genes
class testRandGenes(unittest.TestCase):

    def setUp(self):
        dgeann.random.seed("vigor")

    def test_create_rweights(self):
        lay = [lg(1, False, False, 0, "a", [], 2, "input"),
               lg(1, False, False, 0, "h", ["a"], 2, "IP")]
        #simple case
        simp_genome = dgeann.Genome(lay, lay, [], [])
        net = simp_genome.build().net
        key = "h"
        d = net.params[key][0].data
        simp_genome.weightchr_a = []
        simp_genome.weightchr_b = []
        off = simp_genome.create_rweights("a", d, "h", net)
        self.assertEqual(off, 2)
        self.assertEqual(len(simp_genome.weightchr_a), 4)
        self.assertEqual(len(simp_genome.weightchr_b), 4)
        i = 0
        j = 0
        for gen in simp_genome.weightchr_a:
            self.assertEqual(gen.in_node, j)
            self.assertEqual(gen.out_node, i)
            a = d[i][j]
            self.assertAlmostEqual(gen.weight, a)
            i += 1
            if i == 2:
                i = 0
                j += 1

    def test_concat_rweights(self):
        #concats case
        layers = [lg(1, False, False, 0, "i", [], 1, "input"),
          lg(1, False, False, 0, "a", ["i"], 2, "IP"),
          lg(1, False, False, 0, "b", ["i"], 2, "IP"),
          lg(1, False, False, 0, "c", ["i"], 2, "IP"),
          lg(1, False, False, 0, "d", ["i"], 2, "IP"),
          lg(1, False, False, 0, "h", ["a", "b", "c", "d"], 2, "IP")]
        concats_genome = dgeann.Genome(layers, layers, [], [])
        key = "h"
        concat_dict = {"ZPKRMC": [["a", "b", "c", "d"], [2, 2, 2, 2], ["h"]]}
        net =concats_genome.build().net
        d = net.params[key][0].data
        self.assertEqual(len(concats_genome.weightchr_a), 24)
        i = 0
        j = 0
        off_dict = {"a": 0, "b": 2, "c": 4, "d": 6}
        for gen in concats_genome.weightchr_a[8:]:
            self.assertEqual(gen.in_node, j)
            self.assertEqual(gen.out_node, i)
            a = d[i][j+off_dict[gen.in_layer]]
            self.assertAlmostEqual(gen.weight, a)
            i += 1
            if i == 2:
                i = 0
                j += 1
            if j == 2:
                j = 0

#tests the genome mutation functions
class testMutation(unittest.TestCase):

    def setUp(self):
        dgeann.random.seed("vigor")

    def test_gene_ident(self):
        ident = dgeann.gene_ident()
        self.assertEqual(ident, "QXLPYG")

    def test_new_input(self):
        test_layer_b = lg(4, True, True, .01, "INa", [], 5, "IP")
        test_dup_layer = lg(4, True, True, .01, "ABCDEF", [], 5, "IP")
        test_genome = dgeann.Genome([test_dup_layer, test_layer_b],
                                    [test_layer_b], [],[])
        x = test_genome.new_input(test_genome.layerchr_a[0],
                                  test_genome.layerchr_a)
        self.assertEqual(x.ident, "INa")
        #test with multiple possible layers
        test_layer_c = lg(4, True, True, .01, "INi", [], 5, "IP")
        test_layer_d = lg(4, True, True, .01, "INu", [], 5, "IP")
        test_genome_b = dgeann.Genome([test_layer_b, test_layer_c,
                                       test_layer_d],
                                      [test_dup_layer, test_layer_b,
                                       test_layer_c, test_layer_d], [], [])
        x = test_genome_b.new_input(test_genome_b.layerchr_b[0],
                                    test_genome_b.layerchr_b)
        
        self.assertEqual("INi", x.ident)
        #test with concats and loss layer at end
        test_loss_layer = lg(4, True, True, .01, "loss", [], 5, "loss")
        test_genome_c = dgeann.Genome([test_layer_b,  test_loss_layer],
                                        [test_dup_layer, test_layer_b, 
                                         test_loss_layer], [], [])
        x = test_genome_c.new_input(test_genome_c.layerchr_b[0],
                                    test_genome_c.layerchr_b)
        self.assertEqual(x.ident, "INa")

    def test_find_n_inputs(self):
        #single input
        test_in = lg(1, False, False, 0, "IN", [], 5, "data")
        test_out = lg(1, False, False, 0, "OUT", ["IN"], 5, "IP")
        genome = dgeann.Genome([test_in, test_out], [], [], [])
        n, d = genome.find_n_inputs(test_out, genome.layerchr_a)
        self.assertEqual(n, 5)
        self.assertEqual({"IN": 5}, d)
        #concat input
        test_in2 = lg(1, False, False, 0, "IN2", [], 5, "data")
        test_out = lg(1, False, False, 0, "OUT", ["IN", "IN2"], 5, "IP")
        genome2 = dgeann.Genome([test_in, test_in2, test_out], [], [], [])
        n, d = genome.find_n_inputs(test_out, genome2.layerchr_a)
        self.assertEqual(n, 10)
        self.assertEqual({"IN": 5, "IN2": 5}, d)

    def test_dup_weights(self):
        #simplest test case
        test_in = lg(4, False, False, 0, "IN", [], 1, "data")
        test_layer_b = lg(4, True, True, .01, "OUT", ["ABCDEF"], 5, "IP")
        test_dup_layer = lg(4, True, True, .01, "ABCDEF", ["IN"], 5, "IP")
        test_genome = dgeann.Genome([test_in, test_dup_layer, test_layer_b], [],
                                      [], [])
        test_genome.dup_weights(test_dup_layer, test_layer_b,
                                test_genome.layerchr_a)
        self.assertEqual(len(test_genome.weightchr_a), 30)
        n = 0
        m = 0
        o = 0
        for g in test_genome.weightchr_a:
            self.assertEqual(g.in_node, m)
            self.assertEqual(g.out_node, n)
            if o == 1:
                self.assertEqual(g.in_layer, "ABCDEF")
                self.assertEqual(g.out_layer, "OUT")
            else:
                self.assertEqual(g.in_layer, "IN")
                self.assertEqual(g.out_layer, "ABCDEF")
            n += 1
            if o == 0 and n == 5:
                o = 1
                n = 0
            else:
                if n == 5:
                    n = 0
                    m += 1

    def test_handle_duplication(self):
        test_input = lg(4, False, False, 0, "d", [], 1, "data")
        test_layer_b = lg(4, True, True, .01, "INa", ["d"], 5, "IP")
        test_genome = dgeann.Genome([test_input, test_layer_b], [],
                                      [], [])
        test_genome.handle_duplication(test_layer_b, test_genome.layerchr_a)
        self.assertEqual(len(test_genome.layerchr_a), 3)
        self.assertNotEqual(test_genome.layerchr_a[1].ident, "INa")
        self.assertEqual(len(test_genome.weightchr_a), 30)
        self.assertEqual(len(test_genome.weightchr_b), 30)
        i = 0
        j = 0
        k = 0
        for gene in test_genome.weightchr_a:
            if k == 0:
                self.assertEqual(gene.in_layer, "d")
            else:
                self.assertEqual(gene.out_layer, "INa")
            self.assertEqual(gene.out_node, i)
            self.assertEqual(gene.in_node, j)
            i += 1
            if i > 4:
                i = 0
                if k == 0:
                    k = 1
                else:
                    j += 1

    def test_find_outputs(self):
        #simple case
        test_in = lg(5, False, False, 0, "d", [], 1, "data")
        test_gene = lg(5, True, True, .017, "tester", ["d"], 5, "IP")
        test_genome = dgeann.Genome([test_in, test_gene], [], [], [])
        out = test_genome.find_outputs(test_in, test_genome.layerchr_a)
        self.assertEqual(out, [test_gene])
        #case with 3 outputs across two concats
        conc1 = lg(5, False, False, 0, "con1", ["tester"], None, "concat")
        conc2 = lg(5, False, False, 0, "con2", ["tester", "d"], None, "concat")
        out1 = lg(5, True, True, .017, "out1", ["con1"], 5, "IP")
        out2 = lg(5, True, True, .017, "out2", ["con2"], 3, "IP")
        out3 = lg(5, True, True, .017, "out3", ["con2"], 6, "IP")
        test_genome2 = dgeann.Genome([test_in, test_gene, conc1, conc2, out1,
                                        out2, out3], [], [], [])
        out = test_genome2.find_outputs(test_gene, test_genome2.layerchr_a)
        self.assertEqual(out, [out1, out2, out3])

    def test_add_nodes(self):
        test_in = lg(5, False, False, 0, "d", [], 1, "data")
        test_gene = lg(5, True, True, .01, "tester", ["d"], 5, "IP")
        test_out = lg(5, False, False, 0, "o", ["tester"], 3, "IP")
        weights = []
        for i in range(5):
            w = wg(5, True, False, .01, str(i), 3, 0, i, "d", "tester")
            weights.append(w)
        for i in range(5):
            for j in range (3):
                w = wg(5, True, False, .01, str(i), 3, i, j, "tester", "o")
                weights.append(w)
        test_genome = dgeann.Genome([test_in, test_gene, test_out], [],
                                      weights, [])
        n_in, d = test_genome.find_n_inputs(test_gene, test_genome.layerchr_a)
        test_genome.add_nodes(test_gene, test_genome.layerchr_a, 2,
                              test_genome.weightchr_a, n_in)
        test_gene.nodes += 2
        self.assertEqual(len(test_genome.weightchr_a), 28)
        self.assertEqual(len(test_genome.weightchr_b), 0)
        #test that inputs/outputs are right
        n = 0
        for g in test_genome.weightchr_a[:6]:
            self.assertEqual(g.out_node, n)
            n += 1
        n = 0
        m = 0
        for g in test_genome.weightchr_a[7:]:
            self.assertEqual(g.in_node, n)
            self.assertEqual(g.out_node, m)
            m += 1
            if m > 2:
                m = 0
                n += 1

    def test_add_nodes_complicated(self):
        #more complicated version: 3 outputs on 2 concats, weights on both chrs
        out1 = lg(5, True, True, .017, "out1", ["tester"], 5, "IP")
        out2 = lg(5, True, True, .017, "out2", ["tester", "d"], 3, "IP")
        out3 = lg(5, True, True, .017, "out3", ["tester", "d"], 6, "IP")
        test_gene = lg(5, True, True, .01, "tester", ["d"], 3, "IP")
        test_in = lg(5, False, False, 0, "d", [], 1, "data")
        test_out = lg(5, False, False, 0, "o", ["tester"], 3, "IP")
        weights = []
        compare_weights = []
        for i in range(4):
            w = wg(5, True, False, .01, str(i), 3, 0, i, "d", "tester")
            if i < 3:
                weights.append(w)
            compare_weights.append(w)
        for i in range(4):
            for j in range(5):
                w = wg(5, True, False, .01, str(i), 3, i, j, "tester", "out1")
                if i < 3:
                    weights.append(w)
                compare_weights.append(w)
        for i in range(4):
            for j in range (3):
                w = wg(5, True, False, .01, str(i), 3, i, j, "tester", "out2")
                if i < 3:
                    weights.append(w)
                compare_weights.append(w)
        for i in range(4):
            for j in range (6):
                w = wg(5, True, False, .01, str(i), 3, i, j, "tester", "out3")
                if i < 3:
                    weights.append(w)
                compare_weights.append(w)
        test_genome = dgeann.Genome([test_in, test_gene, test_out], [],
                                      weights, [])
        test_genome2 = dgeann.Genome([test_in, test_gene, out1, out2, out3],
                                     [], weights, weights)
        n_in, d = test_genome.find_n_inputs(test_gene, test_genome.layerchr_a)
        test_genome2.add_nodes(test_gene, test_genome2.layerchr_a, 1,
                               test_genome2.weightchr_a, n_in)
        test_genome2.add_nodes(test_gene, test_genome2.layerchr_b, 1,
                               test_genome2.weightchr_b, n_in)
        test_gene.nodes += 1     
        #45 original + 1 + 5 + 3 + 6 = 60
        self.assertEqual(len(test_genome2.weightchr_a), 60)
        self.assertEqual(len(test_genome2.weightchr_b), 60)
        for i in range(len(compare_weights)):
            self.assertEqual(compare_weights[i].in_node,
                             test_genome2.weightchr_a[i].in_node)
            self.assertEqual(compare_weights[i].in_node,
                             test_genome2.weightchr_b[i].in_node)
            self.assertEqual(compare_weights[i].out_node,
                             test_genome2.weightchr_a[i].out_node)
            self.assertEqual(compare_weights[i].out_node,
                             test_genome2.weightchr_b[i].out_node)
        #now a case where the layer shrank in the past, then re-expands
        test_gene.nodes -= 2
        n_in, d = test_genome.find_n_inputs(test_gene, test_genome.layerchr_a)
        test_genome2.add_nodes(test_gene, test_genome2.layerchr_a, 2,
                               test_genome2.weightchr_a, n_in)
        test_genome2.add_nodes(test_gene, test_genome2.layerchr_b, 2,
                               test_genome2.weightchr_b, n_in)
        test_gene.nodes += 2
        self.assertEqual(len(test_genome2.weightchr_a), 60)
        self.assertEqual(len(test_genome2.weightchr_b), 60)
        for i in range(len(compare_weights)):
            self.assertEqual(compare_weights[i].in_node,
                             test_genome2.weightchr_a[i].in_node)
            self.assertEqual(compare_weights[i].in_node,
                             test_genome2.weightchr_b[i].in_node)
            self.assertEqual(compare_weights[i].out_node,
                             test_genome2.weightchr_a[i].out_node)
            self.assertEqual(compare_weights[i].out_node,
                             test_genome2.weightchr_b[i].out_node)
        #and lastly a case where the layer shrank, but expanded to become
        #larger than it originally was
        test_gene.nodes -= 2
        n_in, d = test_genome.find_n_inputs(test_gene, test_genome.layerchr_a)
        test_genome2.add_nodes(test_gene, test_genome2.layerchr_a, 3,
                               test_genome2.weightchr_a, n_in)
        test_genome2.add_nodes(test_gene, test_genome2.layerchr_b, 3,
                               test_genome2.weightchr_b, n_in)
        test_gene.nodes +=3
        self.assertEqual(len(test_genome2.weightchr_a), 75)
        self.assertEqual(len(test_genome2.weightchr_b), 75)
        compare_weights = []
        for i in range(5):
            w = wg(5, True, False, .01, str(i), 3, 0, i, "d", "tester")
            compare_weights.append(w)
        for i in range(5):
            for j in range(5):
                w = wg(5, True, False, .01, str(i), 3, i, j, "tester", "out1")
                compare_weights.append(w)
        for i in range(5):
            for j in range (3):
                w = wg(5, True, False, .01, str(i), 3, i, j, "tester", "out2")
                compare_weights.append(w)
        for i in range(5):
            for j in range (6):
                w = wg(5, True, False, .01, str(i), 3, i, j, "tester", "out3")
                compare_weights.append(w)
        for i in range(len(compare_weights)):
            self.assertEqual(compare_weights[i].in_node,
                             test_genome2.weightchr_a[i].in_node)
            self.assertEqual(compare_weights[i].in_node,
                             test_genome2.weightchr_b[i].in_node)
            self.assertEqual(compare_weights[i].out_node,
                             test_genome2.weightchr_a[i].out_node)
            self.assertEqual(compare_weights[i].out_node,
                             test_genome2.weightchr_b[i].out_node)    
        
    def test_handle_mutation_weights(self):
        test_weight_a = wg(5, True, False, 0, "au00", 3.00, 0, 0, "INa", "IPu")
        test_weight_b = wg(4, True, False, 1.0, "au00", 3.00, 0, 0,
                           "INa", "IPu")
        test_layer_a = lg(5, True, False, 0, "INa", [], 5, "input")
        test_layer_b = lg(4, True, True, .01, "INa", [], 5, "IP")
        test_genome = dgeann.Genome([test_layer_a], [test_layer_b],
                                      [test_weight_a], [test_weight_b])
        test_genome.handle_mutation("Dom, -1", test_weight_a, "a")
        self.assertEqual(test_weight_a.dom, 4)
        test_genome.handle_mutation("Dom, 1", test_weight_b, "b")
        self.assertEqual(test_weight_b.dom, 5)
        test_genome.handle_mutation("Weight, 3.0", test_weight_a, "a")
        self.assertEqual(test_weight_a.weight, 6.00)
        test_genome.handle_mutation("Weight, -3.0", test_weight_b, "b")
        self.assertEqual(test_weight_b.weight, 0.00)
        test_genome.handle_mutation("Rate, 1", test_weight_a, "a")
        self.assertEqual(test_weight_a.mut_rate, 1)
        test_genome.handle_mutation("Rate, -1", test_weight_a, "a")
        self.assertEqual(test_weight_a.mut_rate, 0)
        test_genome.handle_mutation("Rate, -1", test_weight_b, "b")
        self.assertEqual(test_weight_b.mut_rate, 0)
        test_genome.handle_mutation("Rate, 1", test_weight_b, "b")
        self.assertEqual(test_weight_b.mut_rate, 1.0)

    def test_handle_mutation_layers(self):
        test_weight_a = wg(5, True, False, 0, "au00", 3.00, 0, 0, "INa", "IPu")
        test_weight_b = wg(4, True, False, 1.0, "au00", 3.00, 0, 0,
                           "INa", "IPu")
        test_layer_a = lg(5, True, False, 0, "INa", [], 5, "input")
        test_layer_b = lg(4, True, True, .01, "INa", [], 5, "IP")
        test_genome = dgeann.Genome([test_layer_a], [test_layer_b],
                                      [test_weight_a], [test_weight_b])
        test_genome.handle_mutation("Rate, 1", test_layer_a, "a")
        self.assertEqual(test_layer_a.mut_rate, 1)
        test_genome.handle_mutation("Rate, -1", test_layer_a, "a")
        self.assertEqual(test_layer_a.mut_rate, 0)
        test_genome.handle_mutation("Rate, -.001", test_layer_b, "b")
        self.assertAlmostEqual(test_layer_b.mut_rate, .009)
        test_genome.handle_mutation("Dom, -1", test_layer_a, "a")
        self.assertEqual(test_layer_a.dom, 4)
        test_genome.handle_mutation("Dom, 1", test_layer_b, "b")
        self.assertEqual(test_layer_b.dom, 5)
        #test duplication
        test_in = lg(5, False, False, 0, "d", [], 1, "data")
        test_dupl = lg(5, True, True, .017, "TEST", ["d"], 5, "IP")
        test_dup = dgeann.Genome([], [test_in, test_dupl], [], [])
        test_dup.handle_mutation("Duplicate,", test_dup.layerchr_b[1], "b",
                                 test_dup.layerchr_b)
        self.assertEqual(len(test_dup.layerchr_b), 3)
        self.assertEqual(len(test_dup.layerchr_b[1].inputs), 1)
        self.assertEqual(len(test_dup.layerchr_b[2].inputs), 2)
        self.assertEqual(len(test_dup.weightchr_b), 30)
        self.assertEqual(len(test_dup.weightchr_a), 30)
        #test change in node #
        #first losing nodes
        test_dup.handle_mutation("Nodes, -3", test_dup.layerchr_b[1], "b",
                                 test_dup.layerchr_b)
        self.assertEqual(len(test_dup.weightchr_b), 30)
        self.assertEqual(test_dup.layerchr_b[1].nodes, 2)
        #now adding nodes to one that used to have them and don't need more
        test_dup.handle_mutation("Nodes, 1", test_dup.layerchr_b[1], "b",
                                test_dup.layerchr_b)
        self.assertEqual(test_dup.layerchr_b[1].nodes, 3)
        self.assertEqual(len(test_dup.weightchr_b), 30)
        #and adding nodes to one that did not used to have them
        test_dup.handle_mutation("Nodes, 3", test_dup.layerchr_b[1], "b",
                                test_dup.layerchr_b)
        self.assertEqual(test_dup.layerchr_b[1].nodes, 6)
        self.assertEqual(len(test_dup.weightchr_b), 36)
        #more complicated case where different numbers of weights are needed
        #for different layers
        test_dup.layerchr_b.append(lg(5, False, False, 3, "TEST2",
                                      ["QXLPYG"], 1, "IP"))
        for i in range(7):
            test_dup.weightchr_b.append(wg(5, True, True, .01, str(i), 3, i,
                                           0, "QXLPYG", "TEST2"))
        test_dup.handle_mutation("Nodes, 2", test_dup.layerchr_b[1], "b",
                                test_dup.layerchr_b)
        self.assertEqual(test_dup.layerchr_b[1].nodes, 8)
        self.assertEqual(len(test_dup.weightchr_b), 56)
        
    def test_mutate(self):
        dgeann.random.seed("genetic1")
        lgene_0 = lg(1, True, False, 1, "a", [], 5, "IP")
        wgene_0 = wg(5, True, False, 1.0, "au00", 3.00, 0, 0, "INa", "IPu")
        wgene_1 = wg(5, True, False, 1.0, "au00", 3.00, 0, 0, "INa", "IPu")
        wgene_2 = wg(5, True, False, .066, "au00", 3.00, 0, 0, "INa", "IPu")
        wgene_3 = wg(5, False, False, 1.0, "au00", 3.00, 0, 0, "INa", "IPu")
        wgene_4 = wg(5, True, False, 1.0, "au04", 3.00, 0, 0, "INa", "IPu")
        wgene_5 = wg(5, True, False, 1.0, "au05", 3.00, 0, 0, "INa", "IPu")
        test_genome = dgeann.Genome([lgene_0], [lgene_0],
                                    [wgene_0, wgene_1, wgene_2],
                                    [wgene_3, wgene_4, wgene_5])
        test_genome.mutate()
        self.assertNotEqual(wgene_0.weight, 3.00)
        self.assertNotEqual(wgene_1.weight, 3.00)
        self.assertEqual(wgene_2.weight, 3.00)
        self.assertEqual(wgene_2.dom, 5)
        self.assertEqual(wgene_2.mut_rate, 0.066)
        self.assertEqual(wgene_3.weight, 3.00)
        self.assertEqual(wgene_3.dom, 5)
        self.assertEqual(wgene_3.mut_rate, 1.0)
        self.assertNotEqual(wgene_4.weight, 3.00)
        self.assertNotEqual(wgene_5.dom, 5)
        

#tests recombination and helper functions
class testRecombination(unittest.TestCase):

    def setUp(self):
        dgeann.random.seed("genetic")
        #simple case
        layer_a = lg(5, False, False, 0, "INa", [], 5, "input")
        layer_i = lg(5, False, False, 0, "INi", [], 5, "input")
        layer_concat = lg(5, False, False, 0, "concat",
                          ["INa", "INi"], None, "concat")
        layer_u = lg(5, False, False, 0, "IPu", ["concat"], 5, "IP")
        layer_o = lg(5, False, False, 0, "IPo", ["concat"], 5, "IP")
        layer_list_a = [layer_a, layer_i, layer_concat, layer_u, layer_o]
        layer_a2 = lg(5, False, False, 0, "INa", [], 5, "input")
        layer_o2 = lg(5, False, False, 0, "IPo", ["concat"], 5, "IP")
        layer_list_b = [layer_a2, layer_i, layer_concat, layer_u, layer_o2]
        a_u_00 = wg(5, False, False, 0, "au00", 3.00, 0, 0, "INa", "IPu")
        a_o_00 = wg(5, False, False, 0, "ao00", 3.00, 0, 0, "INa", "IPo")
        i_u_00 = wg(5, False, False, 0, "iu00", 5.00, 0, 0, "INi", "IPu")
        i_o_00 = wg(5, False, False, 0, "io00", 5.00, 0, 0, "INi", "IPo")
        weight_list_a = [a_u_00, a_o_00, i_u_00, i_o_00]
        a_u = wg(5, False, False, 0, "au00", 7.00, 0, 0, "INa", "IPu")
        i_o = wg(5, False, False, 0, "io00", 7.00, 0, 0, "INi", "IPo")
        weight_list_b = [a_u, a_o_00, i_u_00, i_o]
        self.genome_a = dgeann.Genome(layer_list_a, layer_list_b,
                                   weight_list_a, weight_list_b)
        #a case with more complicated weight genetics (extras in front) 
        a_u_01 = wg(5, False, False, 0, "au01", 3.00, 0, 1, "INa", "IPu")
        a_u_02 = wg(5, False, False, 0, "au02", 3.00, 0, 2, "INa", "IPu")
        a_u_03 = wg(5, False, False, 0, "au03", 3.00, 0, 3, "INa", "IPu")
        a_o_01 = wg(5, False, False, 0, "ao01", 3.00, 0, 3, "INa", "IPo")
        weight_list_c = [a_u_00, a_u_01, a_u_02, a_u_03, a_o_00, a_o_01, i_u_00,
                         i_o_00]
        weight_list_d = [a_u_00, a_u_01, a_o_00, a_o_01, i_u_00, i_o_00]
        self.genome_b = dgeann.Genome(layer_list_a, layer_list_b, weight_list_c,
                                        weight_list_d)
        #and switched around
        self.genome_c = dgeann.Genome(layer_list_a, layer_list_b, weight_list_d,
                                        weight_list_c)
        #case where the extras are on the end
        #and one strand gets an extra layer
        i_o_01 = wg(5, False, False, 0, "io00", 5.00, 0, 1, "INi", "IPo")
        i_o_02 = wg(5, False, False, 0, "io00", 5.00, 0, 2, "INi", "IPo")
        layer_e = lg(5, False, False, 0, "IPe", ["concat"], 5, "IP")
        layer_list_c = [layer_a, layer_i, layer_concat, layer_u,
                        layer_o, layer_e]
        weight_list_e = [a_u_00, a_u_01, a_o_00, a_o_01, i_u_00, i_o_00, i_o_01,
                         i_o_02]
        self.genome_d = dgeann.Genome(layer_list_c, layer_list_b, weight_list_e,
                                        weight_list_d)
        self.genome_e = dgeann.Genome(layer_list_a, layer_list_c, weight_list_d,
                                        weight_list_e)
        #case where the extras are in the middle
        i_u_01 = wg(5, False, False, 0, "iu01", 5.00, 0, 1, "INi", "IPu")
        i_u_02 = wg(5, False, False, 0, "iu02", 5.00, 0, 2, "INi", "IPu")
        weight_list_f = [a_u_00, a_u_01, a_o_00, a_o_01, i_u_00, i_u_01, i_u_02,
                         i_o_00]
        self.genome_f = dgeann.Genome(layer_list_a, layer_list_b, weight_list_f,
                                        weight_list_d)
        self.genome_g = dgeann.Genome(layer_list_a, layer_list_b, weight_list_d,
                                        weight_list_f)
        #case where the extras are for a different layer, at end
        other_0 = wg(5, False, False, 0, "asdl", 5.00, 0, 0, "asdl", "jkl;")
        other_1 = wg(5, False, False, 0, "jkl;", 5.00, 0, 1, "asdl", "jkl;")
        weight_list_g = [a_u_00, a_u_01, a_o_00, a_o_01, i_u_00, i_o_00,
                         other_0, other_1]
        self.genome_h = dgeann.Genome(layer_list_a, layer_list_b, weight_list_g,
                                        weight_list_d)
        self.genome_i = dgeann.Genome(layer_list_a, layer_list_b, weight_list_d,
                                        weight_list_g)
        #case where the extras are for a different layer (both chromosomes)
        #and there's non-matching extra as the end of the layer chromosomes
        other_2 = wg(5, False, False, 0, "zxcv", 5.00, 0, 0, "zxcv", "bnm,")
        other_3 = wg(5, False, False, 0, "bnm,", 5.00, 0, 1, "zxcv", "bnm,")
        layer_ka = lg(5, False, False, 0, "INka",[], 5, "input")
        layer_list_d = [layer_a, layer_i, layer_concat, layer_u, layer_o,
                        layer_ka]
        weight_list_i = [a_u_00, a_u_01, a_o_00, a_o_01, i_u_00, i_o_00,
                         other_2, other_3]
        self.genome_j = dgeann.Genome(layer_list_c, layer_list_d, weight_list_i,
                                        weight_list_g)
        self.genome_k = dgeann.Genome(layer_list_d, layer_list_c, weight_list_g,
                                        weight_list_i)

    def test_last_shared(self):
        n, m = self.genome_a.last_shared()
        self.assertEqual(n, 4)
        self.assertEqual(m, 3)
        n, m = self.genome_b.last_shared()
        self.assertEqual(n, 4)
        self.assertEqual(m, 5)
        n, m = self.genome_c.last_shared()
        self.assertEqual(n, 4)
        self.assertEqual(m, 5)
        n, m = self.genome_d.last_shared()
        self.assertEqual(n, 4)
        self.assertEqual(m, 5)
        n, m = self.genome_e.last_shared()
        self.assertEqual(m, 5)
        n, m = self.genome_f.last_shared()
        self.assertEqual(m, 5)
        n, m = self.genome_g.last_shared()
        self.assertEqual(m, 5)
        n, m = self.genome_h.last_shared()
        self.assertEqual(m, 5)
        n, m = self.genome_i.last_shared()
        self.assertEqual(m, 5)
        n, m = self.genome_j.last_shared()
        self.assertEqual(n, 4)
        self.assertEqual(m, 5)
        n, m = self.genome_k.last_shared()
        self.assertEqual(n, 4)
        self.assertEqual(m, 5)
        
    def test_crossover(self):
        #simple test case
        cross_a = self.genome_a.crossover()
        #will cross over at 3, 2
##        self.assertEqual(cross_a.layerchr_a[4].nodes, 5)
##        self.assertEqual(cross_a.layerchr_b[4].nodes, 7)
        self.assertEqual(cross_a.weightchr_a[3].weight, 7.00)
        self.assertEqual(cross_a.weightchr_b[3].weight, 5.00)
        #crossover at 1, 5
        cross_b = self.genome_b.crossover()
        self.assertEqual(len(cross_b.weightchr_b), 8)
        self.assertEqual(len(cross_b.weightchr_a), 6)
        self.assertEqual(cross_b.weightchr_a[5].ident, "io00")
        self.assertEqual(cross_b.weightchr_b[5].ident, "ao01")
        #and other way around
        cross_c = self.genome_c.crossover()
        self.assertEqual(len(cross_c.weightchr_a), 8)
        self.assertEqual(len(cross_c.weightchr_b), 6)
        cross_d = self.genome_d.crossover()
        self.assertEqual(len(cross_d.layerchr_b), 6)
        self.assertEqual(len(cross_d.layerchr_a), 5)
        self.assertEqual(len(cross_d.weightchr_a), 6)
        self.assertEqual(len(cross_d.weightchr_b), 8)
        cross_e = self.genome_e.crossover()
        self.assertEqual(len(cross_e.layerchr_a), 6)
        self.assertEqual(len(cross_e.layerchr_b), 5)
        self.assertEqual(len(cross_e.weightchr_b), 6)
        self.assertEqual(len(cross_e.weightchr_a), 8)
        #crossover at 2, 2
        cross_f = self.genome_f.crossover()
        self.assertEqual(cross_f.weightchr_b[5].ident, "iu01")
        self.assertEqual(cross_f.weightchr_a[5].ident, "io00")
        #crossover at 1, 4
        cross_g = self.genome_g.crossover()
        self.assertEqual(cross_g.weightchr_a[4].ident, "iu00")
        self.assertEqual(cross_g.weightchr_b[4].ident, "iu00")
        self.assertEqual(cross_g.weightchr_a[5].ident, "iu01")
        self.assertEqual(cross_g.weightchr_b[5].ident, "io00")
        cross_h = self.genome_h.crossover()
        self.assertEqual(len(cross_h.weightchr_a), 6)
        self.assertEqual(len(cross_h.weightchr_b), 8)

    def test_crossover_unconstrained(self):
        dgeann.constrain_crossover = False
        cross_a = self.genome_a.crossover()
        #will cross over at 3, 2 again
##        self.assertEqual(cross_a.layerchr_a[4].nodes, 7)
##        self.assertEqual(cross_a.layerchr_b[4].nodes, 5)
        self.assertEqual(cross_a.weightchr_a[3].weight, 7.00)
        self.assertEqual(cross_a.weightchr_b[3].weight, 5.00)
        dgeann.constrain_crossover = True

    def test_crossover_unequal(self):
        #simpler test case
        dat = lg(3, False, False, 0, "dat", [], 1, "input")
        laya2 = lg(3, False, False, 0, "A", ["dat"], 2, "IP")
        laya3 = lg(3, False, False, 0, "A", ["dat"], 3, "IP")
        layb2 = lg(3, False, False, 0, "B", ["A"], 2, "IP")
        layb3 = lg(3, False, False, 0, "B", ["A"], 3, "IP")
        weights_0 = []
        for i in range(3):
            for j in range(2):
                weights_0.append(wg(3, True, False, 0.1, str(i)+str(j),
                                    3.0, i, j, "A", "B"))
        weights_1 = []
        for i in range(2):
            for j in range(3):
                weights_1.append(wg(3, True, False, 0.1, str(i)+str(j),
                                    3.0, i, j, "A", "B"))
        gen2_3 = dgeann.Genome([dat, laya2, layb3], [dat, laya2, layb3],
                               weights_0, weights_0)
        gen3_2 = dgeann.Genome([dat, laya3, layb2], [dat, laya3, layb2],
                               weights_1, weights_1)
        child = gen2_3.recombine(gen3_2)
        child_c = child.crossover()
        self.assertEqual(len(child_c.weightchr_a), 9)
        in_node = 0
        out_node = 0
        for i in child_c.weightchr_a:
            self.assertEqual(i.in_node, in_node)
            self.assertEqual(i.out_node, out_node)
            if out_node < 2:
                out_node += 1
            else:
                in_node += 1
                out_node = 0
        self.assertEqual(len(child_c.weightchr_b), 9)
        in_node = 0
        out_node = 0
        for i in child_c.weightchr_a:
            self.assertEqual(i.in_node, in_node)
            self.assertEqual(i.out_node, out_node)
            if out_node < 2:
                out_node += 1
            else:
                in_node += 1
                out_node = 0
        #test case from actual data that caused bug
        dgeann.random.seed("genetic")
        data = lg(5, False, False, 0, "data", [], 12, "input")
        stm_input = lg(5, False, False, 0, "stm_input", [], 6, "input")
        stm = lg(5, False, False, 0, "STM",["stm_input"], 6, "STMlayer")
        evo = lg(3, True, False, 0.1, "evo", ["data", "STM"], 5, "IP")
        evo2 = lg(3, True, False, 0.1, "evo", ["data", "STM"], 6, "IP")
        action = lg(5, False, False, 0, "action", ["evo"], 6, "IP")
        layers = [data, stm_input, stm, evo, action]
        layers2 = [data, stm_input, stm, evo2, action]
        weights_a = []
        weights_b = []
        for i in range(12):
            for j in range(6):
                weights_a.append(wg(3, True, False, 0.1, str(i)+str(j),
                                    3.0, i, j, "data", "evo"))
        for i in range(6):
            for j in range(6):
                weights_a.append(wg(3, True, False, 0.1, str(i)+str(j),
                                    3.0, i, j, "STM", "evo"))
        for i in range(6):
            for j in range(6):
                weights_a.append(wg(3, True, False, 0.1, str(i)+str(j),
                                    3.0, i, j, "evo", "action"))
        for i in range(12):
            for j in range(5):
                weights_b.append(wg(3, True, False, 0.1, str(i)+str(j),
                                    3.0, i, j, "data", "evo"))
        for i in range(6):
            for j in range(5):
                weights_b.append(wg(3, True, False, 0.1, str(i)+str(j),
                                    3.0, i, j, "STM", "evo"))
        for i in range(5):
            for j in range(6):
                weights_b.append(wg(3, True, False, 0.1, str(i)+str(j),
                                    3.0, i, j, "evo", "action"))
        uneq = dgeann.Genome(layers2, layers, weights_a, weights_b)
        result = uneq.crossover()
        last_in = 0
        last_out = 0
        i = 0
        for j in range(12):
            for k in range(6):
                self.assertEqual(result.weightchr_a[i].in_node, j)
                self.assertEqual(result.weightchr_a[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result.weightchr_a[i].in_node, j)
                self.assertEqual(result.weightchr_a[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result.weightchr_a[i].in_node, j)
                self.assertEqual(result.weightchr_a[i].out_node, k)
                i += 1
        i = 0
        for j in range(12):
            for k in range(6):
                self.assertEqual(result.weightchr_b[i].in_node, j)
                self.assertEqual(result.weightchr_b[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result.weightchr_b[i].in_node, j)
                self.assertEqual(result.weightchr_b[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result.weightchr_b[i].in_node, j)
                self.assertEqual(result.weightchr_b[i].out_node, k)
                i += 1
        dgeann.random.seed("tako")
        uneq2 = dgeann.Genome(layers2, layers, weights_a, weights_b)
        result2 = uneq2.crossover()
        last_in = 0
        last_out = 0
        i = 0
        for j in range(12):
            for k in range(6):
                self.assertEqual(result2.weightchr_a[i].in_node, j)
                self.assertEqual(result2.weightchr_a[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result2.weightchr_a[i].in_node, j)
                self.assertEqual(result2.weightchr_a[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result2.weightchr_a[i].in_node, j)
                self.assertEqual(result2.weightchr_a[i].out_node, k)
                i += 1
        i = 0
        for j in range(12):
            for k in range(6):
                self.assertEqual(result2.weightchr_b[i].in_node, j)
                self.assertEqual(result2.weightchr_b[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result2.weightchr_b[i].in_node, j)
                self.assertEqual(result2.weightchr_b[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result2.weightchr_b[i].in_node, j)
                self.assertEqual(result2.weightchr_b[i].out_node, k)
                i += 1
        dgeann.random.seed(522273)
        uneq3 = dgeann.Genome(layers2, layers, weights_a, weights_b)
        result3 = uneq3.crossover()
        last_in = 0
        last_out = 0
        i = 0
        for j in range(12):
            for k in range(6):
                self.assertEqual(result3.weightchr_a[i].in_node, j)
                self.assertEqual(result3.weightchr_a[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result3.weightchr_a[i].in_node, j)
                self.assertEqual(result3.weightchr_a[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result3.weightchr_a[i].in_node, j)
                self.assertEqual(result3.weightchr_a[i].out_node, k)
                i += 1
        i = 0
        for j in range(12):
            for k in range(6):
                self.assertEqual(result3.weightchr_b[i].in_node, j)
                self.assertEqual(result3.weightchr_b[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result3.weightchr_b[i].in_node, j)
                self.assertEqual(result3.weightchr_b[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result3.weightchr_b[i].in_node, j)
                self.assertEqual(result3.weightchr_b[i].out_node, k)
                i += 1
        dgeann.random.seed(604122)
        uneq4 = dgeann.Genome(layers2, layers, weights_a, weights_b)
        result4 = uneq4.crossover()
        last_in = 0
        last_out = 0
        i = 0
        for j in range(12):
            for k in range(6):
                self.assertEqual(result4.weightchr_a[i].in_node, j)
                self.assertEqual(result4.weightchr_a[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result4.weightchr_a[i].in_node, j)
                self.assertEqual(result4.weightchr_a[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result4.weightchr_a[i].in_node, j)
                self.assertEqual(result4.weightchr_a[i].out_node, k)
                i += 1
        i = 0
        for j in range(12):
            for k in range(6):
                self.assertEqual(result4.weightchr_b[i].in_node, j)
                self.assertEqual(result4.weightchr_b[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result4.weightchr_b[i].in_node, j)
                self.assertEqual(result4.weightchr_b[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result4.weightchr_b[i].in_node, j)
                self.assertEqual(result4.weightchr_b[i].out_node, k)
                i += 1
        dgeann.random.seed(243883)
        uneq5 = dgeann.Genome(layers2, layers, weights_a, weights_b)
        result5 = uneq2.crossover()
        last_in = 0
        last_out = 0
        i = 0
        for j in range(12):
            for k in range(6):
                self.assertEqual(result5.weightchr_a[i].in_node, j)
                self.assertEqual(result5.weightchr_a[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result5.weightchr_a[i].in_node, j)
                self.assertEqual(result5.weightchr_a[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result5.weightchr_a[i].in_node, j)
                self.assertEqual(result5.weightchr_a[i].out_node, k)
                i += 1
        i = 0
        for j in range(12):
            for k in range(6):
                self.assertEqual(result5.weightchr_b[i].in_node, j)
                self.assertEqual(result5.weightchr_b[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result5.weightchr_b[i].in_node, j)
                self.assertEqual(result5.weightchr_b[i].out_node, k)
                i += 1
        for j in range(6):
            for k in range(6):
                self.assertEqual(result5.weightchr_b[i].in_node, j)
                self.assertEqual(result5.weightchr_b[i].out_node, k)
                i += 1
        
    def test_recomb(self):
        recomb_a = self.genome_a.recombine(self.genome_a)
        x = 0
        for x in range(len(recomb_a.layerchr_a)):
            self.assertEqual(recomb_a.layerchr_a[x].ident,
                             self.genome_a.layerchr_a[x].ident)
        x = 0
        for x in range(len(recomb_a.layerchr_b)):
            self.assertEqual(recomb_a.layerchr_b[x].ident,
                             self.genome_a.layerchr_b[x].ident)
            x = 0
        for x in range(len(recomb_a.weightchr_a)):
            self.assertEqual(recomb_a.weightchr_a[x].ident,
                             self.genome_a.weightchr_a[x].ident)
            x = 0
        for x in range(len(recomb_a.weightchr_b)):
            self.assertEqual(recomb_a.weightchr_b[x].ident,
                             self.genome_a.weightchr_b[x].ident)
        #gets other if-branches
        recomb_b = self.genome_a.recombine(self.genome_a)
        x = 0
        for x in range(len(recomb_b.layerchr_a)):
            self.assertEqual(recomb_b.layerchr_a[x].ident,
                             self.genome_a.layerchr_b[x].ident)
        x = 0
        for x in range(len(recomb_b.layerchr_b)):
            self.assertEqual(recomb_b.layerchr_b[x].ident,
                             self.genome_a.layerchr_a[x].ident)
            x = 0
        for x in range(len(recomb_b.weightchr_a)):
            self.assertEqual(recomb_b.weightchr_a[x].ident,
                             self.genome_a.weightchr_b[x].ident)
            x = 0
        for x in range(len(recomb_b.weightchr_b)):
            self.assertEqual(recomb_b.weightchr_b[x].ident,
                             self.genome_a.weightchr_a[x].ident)
        recomb_c = self.genome_a.recombine(self.genome_k)
        self.assertEqual(len(recomb_c.layerchr_b), 6)
        self.assertEqual(len(recomb_c.layerchr_a), 5)
        self.assertEqual(len(recomb_c.weightchr_a), 4)
        self.assertEqual(len(recomb_c.weightchr_b), 8)
        for gene in recomb_c.weightchr_a:
            for weight in recomb_c.weightchr_b:
                if gene.ident == weight.ident:
                    aru = True
            self.assertTrue(aru)

#while DGEANN is meant to deal with diploidy, haploidy is also an option
#tests haploid cases
class testHaploid(unittest.TestCase):

    def test_recomb(self):
        dgeann.random.seed("genetic")
        layer_a = lg(5, False, False, 0, "A", [], 1, "input")
        layer_b = lg(5, False, False, 0, "B", [], 1, "input")
        layer_c = lg(5, False, False, 0, "C", ["A", "B"], 1, "IP")
        layer_d = lg(5, False, False, 0, "D", ["A", "B"], 1, "IP")
        weight_a = wg(5, False, False, 0, "a", 3.00, 0, 0, "A", "C")
        weight_b = wg(5, False, False, 0, "b", 3.00, 0, 0, "B", "D")
        haploid_a = dgeann.HaploidGenome([layer_a, layer_b, layer_c],
                                         [weight_a])
        haploid_b = dgeann.HaploidGenome([layer_b, layer_b, layer_d],
                                         [weight_b])
        haploid_c = haploid_a.recombine(haploid_b)
        self.assertEqual(len(haploid_c.layerchr_a), 3)
        self.assertEqual(len(haploid_c.layerchr_b), 0)
        self.assertEqual(len(haploid_c.weightchr_a), 1)
        self.assertEqual(len(haploid_c.weightchr_b), 0)
        self.assertEqual(haploid_c.layerchr_a[2].ident, "D")
        self.assertEqual(haploid_c.weightchr_a[0].ident, "a")

    def test_build(self):
        dgeann.random.seed("genetic")
        layer_a = lg(5, False, False, 0, "A", [], 1, "input")
        layer_b = lg(5, False, False, 0, "B", [], 1, "input")
        layer_c = lg(5, False, False, 0, "C", ["A", "B"], 1, "IP")
        weight_a = wg(5, False, False, 0, "a", 3.00, 0, 0, "A", "C")
        haploid_a = dgeann.HaploidGenome([layer_a, layer_b, layer_c],
                                         [weight_a])
        hapa = haploid_a.build()
        self.assertEqual(len(hapa.net.params["C"][0].data[0]), 2)
        self.assertEqual(hapa.net.params["C"][0].data[0][0], 3.0)
        

if __name__ == '__main__':
    unittest.main()
