# core.py
# Authors: Jianyu Yang <Jianyu Yang>
# adapted from code written by Avanti Shrikumar and Jacob Schreiber

import pickle

import numpy as np
import scipy.sparse

from collections import OrderedDict
from . import util


class TrackSet(object):
    def __init__(self, chrom_signals, contrib_scores):
        self.chrom_signals = chrom_signals
        self.contrib_scores = contrib_scores
        self.length = len(chrom_signals[0])

    def create_chromlets(self, chromlets):
        for chromlet in chromlets:
            idx = chromlet.example_idx
            s, e = chromlet.start, chromlet.end

            if chromlet.is_revcomp:
                # reverse the order of the nucleotide
                chromlet.chrom_signals = self.chrom_signals[idx][s:e][::-1, :]
                chromlet.contrib_scores = self.contrib_scores[idx][s:e][::-1, :]
            else:
                chromlet.chrom_signals = self.chrom_signals[idx][s:e]
                chromlet.contrib_scores = self.contrib_scores[idx][s:e]

        return chromlets

class Chromlet(object):
    def __init__(self, example_idx, start, end, is_revcomp):
        self.example_idx = example_idx
        self.start = start
        self.end = end
        self.is_revcomp = is_revcomp

        self.chrom_signals = None
        self.contrib_scores = None

    def __str__(self):
        return ("example:"+str(self.example_idx)
                +",start:"+str(self.start)+",end:"+str(self.end)
                +",rc:"+str(self.is_revcomp))

    def __len__(self):
        return self.end - self.start

    @property
    def string(self):
        return str(self.example_idx)+"_"+str(self.start)+"_"+str(self.end)

    def revcomp(self):
        new_seqlet = Chromlet(
                example_idx=self.example_idx,
                start=self.start, end=self.end,
                is_revcomp=(self.is_revcomp==False))

        new_seqlet.chrom_signals = self.chrom_signals[:, ::-1]
        new_seqlet.contrib_scores = self.contrib_scores[:, ::-1]
        return new_seqlet

    def shift(self, shift_amt):
        return Chromlet(
                example_idx=self.example_idx,
                start=self.start+shift_amt, end=self.end+shift_amt,
                is_revcomp=self.is_revcomp)

    def trim(self, start_idx, end_idx):
        if self.is_revcomp == False:
            new_start = self.start + start_idx 
            new_end = self.start + end_idx
        else:
            new_start = self.end - end_idx
            new_end = self.end - start_idx

        new_seqlet = Chromlet(example_idx=self.example_idx,
            start=new_start, end=new_end, is_revcomp=self.is_revcomp)

        s, e = start_idx, end_idx
        new_seqlet.chrom_signals = self.chrom_signals[s:e]
        new_seqlet.contrib_scores = self.contrib_scores[s:e]
        return new_seqlet


class ChromletSet():
    def __init__(self, seqlets):
        self.chromlets = []
        self.unique_chromlets = {}
        self.length = max([len(chromlet) for chromlet in chromlets])  
        
        #TODO
        self._sequence_sum = np.zeros((self.length, 4), dtype='float')
        self._contrib_sum = np.zeros((self.length, 4), dtype='float')
        self._hypothetical_sum = np.zeros((self.length, 4), dtype='float')

        self.sequence = np.zeros((self.length, 4), dtype='float')
        self.contrib_scores = np.zeros((self.length, 4), dtype='float')
        self.hypothetical_contribs = np.zeros((self.length, 4), dtype='float')

        self.per_position_counts = np.zeros((self.length,))

        for seqlet in seqlets:
            if seqlet.string not in self.unique_seqlets: 
                self._add_seqlet(seqlet=seqlet)
        
        self.subclusters = None
        self.subcluster_to_subpattern = None

    def compute_subpatterns(self, perplexity, n_seeds, n_iterations=-1):
        #this method assumes all the seqlets have been expanded so they
        # all start at 0
        X = util.get_2d_data_from_patterns(self.seqlets)[0]
        X = X.reshape(len(X), -1)
    
        n = len(X)
        n_neighb = min(int(perplexity*3 + 2), len(X))

        affmat_nn, seqlet_neighbors = affinitymat.pairwise_jaccard(X, n_neighb)

        distmat_nn = np.log((1.0/(0.5*np.maximum(affmat_nn, 0.0000001)))-1)
        distmat_nn = np.maximum(distmat_nn, 0.0) #eliminate tiny neg floats

        distmat_sp = scipy.sparse.coo_matrix(
                (np.concatenate(distmat_nn, axis=0),
                 (np.array([i for i in range(len(seqlet_neighbors))
                               for j in seqlet_neighbors[i]]).astype("int"),
                  np.concatenate(seqlet_neighbors, axis=0)) ),
                shape=(len(distmat_nn), len(distmat_nn))).tocsr()

        distmat_sp.sort_indices()

        #do density adaptation
        sp_density_adapted_affmat = affinitymat.NNTsneConditionalProbs( perplexity=perplexity)(affmat_nn, seqlet_neighbors)

        sp_density_adapted_affmat += sp_density_adapted_affmat.T
        sp_density_adapted_affmat /= np.sum(sp_density_adapted_affmat.data)

        #Do Leiden clustering
        self.subclusters = cluster.LeidenCluster(sp_density_adapted_affmat,
            n_seeds=n_seeds, n_leiden_iterations=n_iterations) 

        #this method assumes all the seqlets have been expanded so they
        # all start at 0
        subcluster_to_seqletsandalignments = OrderedDict()
        for seqlet, subcluster in zip(self.seqlets, self.subclusters):
            if (subcluster not in subcluster_to_seqletsandalignments):
                subcluster_to_seqletsandalignments[subcluster] = []
            
            subcluster_to_seqletsandalignments[subcluster].append(seqlet)

        subcluster_to_subpattern = OrderedDict([
            (subcluster, SeqletSet(seqletsandalignments))
            for subcluster,seqletsandalignments in
            subcluster_to_seqletsandalignments.items()])

        #resort subcluster_to_subpattern so that the subclusters with the
        # most seqlets come first
        self.subcluster_to_subpattern = OrderedDict(
            sorted(subcluster_to_subpattern.items(),
                   key=lambda x: -len(x[1].seqlets)))

    def copy(self):
        return SeqletSet(seqlets=[seqlet for seqlet in self.seqlets])

    def trim_to_support(self, min_frac, min_num):
        max_support = max(self.per_position_counts)
        num = min(min_num, max_support*min_frac)
        
        left_idx = 0
        while self.per_position_counts[left_idx] < num:
            left_idx += 1

        right_idx = len(self.per_position_counts)
        while self.per_position_counts[right_idx-1] < num:
            right_idx -= 1
        
        return self.trim_to_idx(start_idx=left_idx, end_idx=right_idx) 

    def trim_to_idx(self, start_idx, end_idx):
        new_seqlets = []
        for seqlet in self.seqlets:
                new_seqlet = seqlet.trim(start_idx=start_idx, end_idx=end_idx)
                new_seqlets.append(new_seqlet)
        return SeqletSet(seqlets=new_seqlets)

    def _add_seqlet(self, seqlet):
        n = len(seqlet)

        self.seqlets.append(seqlet)
        self.unique_seqlets[seqlet.string] = seqlet
        self.per_position_counts[:n] += 1.0 

        ppc = self.per_position_counts[:, None]
        ppc = ppc + 1E-7 * (ppc == 0)

        self._sequence_sum[:n] += seqlet.sequence
        self._contrib_sum[:n] += seqlet.contrib_scores
        self._hypothetical_sum[:n] += seqlet.hypothetical_contribs

        self.sequence = self._sequence_sum / ppc
        self.contrib_scores = self._contrib_sum / ppc
        self.hypothetical_contribs = self._hypothetical_sum / ppc

    def __len__(self):
        return self.length

    def save_seqlets(self, filename):
        bases = np.array(['A', 'C', 'G', 'T'])

        with open(filename, "w") as outfile:
            for seqlet in self.seqlets:
                sequence = "".join(bases[np.argmax(seqlet.sequence, axis=-1)])
                example_index = seqlet.example_idx
                start, end = seqlet.start, seqlet.end
                outfile.write(">example%d:%d-%d\n" % (example_index, start, end))
                outfile.write(sequence + "\n")
