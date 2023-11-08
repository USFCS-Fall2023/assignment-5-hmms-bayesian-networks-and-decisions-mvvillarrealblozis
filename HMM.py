import random
import argparse
import codecs
import os
import sys

import numpy

# observations
import numpy as np


class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):

        with open(f"{basename}.trans", 'r') as F:
            for line in F:
                parts = line.split()
                state_from = parts[0]
                state_to = parts[1]
                probability = float(parts[2])

                # check if state_from is already a key in the transitions dictionary
                if state_from not in self.transitions:
                    self.transitions[state_from] = {}

                # add the transition to the state_from entry in the dictionary
                self.transitions[state_from][state_to] = probability

        with open(f"{basename}.emit", 'r') as F:
            for line in F:
                parts = line.split()
                state = parts[0]
                emission = parts[1]
                probability = float(parts[2])

                # check if state is already a key in the emissions dictionary
                if state not in self.emissions:
                    self.emissions[state] = {}

                # add the emission to the state entry in the dictionary
                self.emissions[state][emission] = probability

    ## you do this.
    def generate(self, n):
        current_state = '#'
        states_seq = []
        outputs_seq = []

        # generate n observations
        for i in range(n):
            # get possible next states and their probabilities
            next_states = []
            next_state_probs = []
            for state, prob in self.transitions[current_state].items():
                next_states.append(state)
                next_state_probs.append(prob)

            # randomly choose the next state
            next_state = np.random.choice(next_states, p=next_state_probs)
            states_seq.append(next_state)

            # randomly choose an emission
            emissions = []
            emissions_probs = []
            for emission, prob in self.emissions[next_state].items():
                emissions.append(emission)
                emissions_probs.append(prob)
            output = np.random.choice(emissions, p=emissions_probs)
            outputs_seq.append(output)

            current_state = next_state

        return Observation(states_seq, outputs_seq)

    def forward(self, observation):
        states = list(self.transitions.keys())
        states.remove('#')
        observation = observation.split()
        observation = observation[:-1]

        T = len(observation)
        N = len(states)

        # init forward probabilities matrix
        M = []
        for _ in range(T):
            M.append([0] * N)

        # init step
        for s, state in enumerate(states):
            M[0][s] = self.transitions['#'][state] * self.emissions[state].get(observation[0], 0)

        # iteration step
        for t in range(1, T):
            for s, state in enumerate(states):
                sum_val = 0
                for s2, prev_state in enumerate(states):
                    sum_val += M[t - 1][s2] * self.transitions[prev_state].get(state, 0) * self.emissions[state].get(
                        observation[t], 0)
                M[t][s] = sum_val

        # termination step
        last_probs = M[-1]
        max_prob = max(last_probs)
        most_likely_state = states[last_probs.index(max_prob)]

        return most_likely_state


    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):

        col_list = observation.outputseq
        row_list = list(self.transitions.keys())
        row_list.remove('#')
        row_list.insert(0, '#')

        col = len(col_list)
        row = len(row_list)

        # create matrices for probabilities and backpointers
        matrix = [[0.0 for _ in range(col)] for _ in range(row)]
        backpointers = [[None for _ in range(col)] for _ in range(row)]

        # init first column of the Viterbi matrix
        for i in range(1, row):
            key = row_list[i]
            if key in self.emissions and col_list[0] in self.emissions[key]:
                matrix[i][0] = self.transitions['#'][key] * self.emissions[key][col_list[0]]
            backpointers[i][0] = 0

        # fill in matrix
        for j in range(1, col):
            for i in range(1, row):
                key = row_list[i]
                if key in self.emissions and col_list[j] in self.emissions[key]:
                    max_prob = -1.0
                    max_state = -1
                    for k in range(1, row):
                        prob = self.emissions[key][col_list[j]] * \
                               self.transitions[row_list[k]][key] * \
                               matrix[k][j - 1]
                        if prob > max_prob:
                            max_prob = prob
                            max_state = k
                    matrix[i][j] = max_prob
                    backpointers[i][j] = max_state

        # find most likely last state
        max_prob = -sys.maxsize - 1
        last_state = None
        for i in range(1, row):
            if matrix[i][-1] > max_prob:
                max_prob = matrix[i][-1]
                last_state = i

        # backtrack to find  most likely states
        most_likely_states = [row_list[last_state]]
        for j in range(col - 1, 0, -1):
            last_state = backpointers[last_state][j]
            most_likely_states.insert(0, row_list[last_state])

        return most_likely_states

