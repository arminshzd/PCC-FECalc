from pathlib import Path
import pickle

import numpy as np
import numpy.typing as npt
import torch
import gpytorch

class GenericStringKernel(gpytorch.kernels.Kernel):

    has_lengthscale = True

    def __init__(self, translator, L: int = 5, **kwargs) -> None:
        super().__init__(ard_num_dims=2, **kwargs)
        self.translator = translator
        self.str2ij = {key: i for i, key in enumerate(self.translator.psi_dict.keys())}
        self.ij2str = {i: key for i, key in enumerate(self.translator.psi_dict.keys())}
        self.E_ij = self.get_Eij()
        self.L = L
        self.t_ij = None # these are functions of the lengthscale
        self.Bij_dict = {}

    def get_E(self, psi_a, psi_ap):
        """
        Calculate E(a, a')

        Args:
            psi_a (torch.tensor): property array of AA a
            psi_ap (torch.tensor): property array of AA a'
        """
        return torch.sum(torch.square(psi_a-psi_ap), dim=0)

    def get_Eij(self):
        E_ij = torch.zeros((len(self.str2ij), len(self.str2ij)))
        for i in range(E_ij.shape[0]):
            for j in range(i, E_ij.shape[1]):
                str_i = self.ij2str[i]
                str_j = self.ij2str[j]
                psi_i = self.translator.psi_dict[str_i].div(torch.norm(self.translator.psi_dict[str_i]))
                psi_j = self.translator.psi_dict[str_j].div(torch.norm(self.translator.psi_dict[str_j]))
                E_ij[i, j] = self.get_E(psi_i, psi_j)
                E_ij[j, i] = E_ij[i, j]
        return E_ij

    def get_tij(self):
        t_ij = torch.zeros_like(self.E_ij)
        for i in range(t_ij.shape[0]):
            for j in range(i, t_ij.shape[1]):
                t_ij[i, j] = torch.exp(-(self.E_ij[i, j]).div(2).div(self.lengthscale[0, 1])) #sigma_c
                t_ij[j, i] = t_ij[i, j]
        return t_ij
    
    def get_dist(self, psi_l_1, psi_l_2):
        """
        Calculate eucledian dist of two epitopes of length l

        Args:
            psi_l_1 (torch.tensor): matrix of size d-by-l for properties of epitope 1
            psi_l_2 (torch.tensor): matrix of size d-by-l for properties of epitope 2
        """
        dist = torch.square(psi_l_1 - psi_l_2)
        return torch.sum(torch.sum(dist, dim=0), dim=0)

    
    def get_Bij_deprecated(self, psi1, psi2, i, j):
        """
        Calcualte B_ij = sum_{l=1}^{min(L, seq1-i, seq2-j)} exp(-sum_{k=1}^l E(x_{i+k}, x'_{j+k})/2/sigma_c^2)
        Args:
            psi1 (torch.Tensor): 24xl Blosum62 representation of sequence 1
            psi2 (torch.Tensor): 24xl Blosum62 representation of sequence 2
            i (int): Shift in sequence 1
            j (int): Shift in sequence 2

        Returns:
            torch.tensor(float): B_ij
        """
        sum_len = min(self.L, psi1.shape[1]-i, psi2.shape[1]-j)
        running_sum = torch.tensor(0.0)
        for l in range(1, sum_len+1):
            sub_psi_1 = psi1[..., i:i+l]
            sub_psi_2 = psi2[..., j:j+l]
            running_sum += torch.exp(-self.get_dist(sub_psi_1, sub_psi_2).div(2).div(self.lengthscale[0, 1]))
        return running_sum

    def get_Bij(self, seq1, seq2):
        """
        Calcualte B_ij = sum_{l=1}^{min(L, seq1-i, seq2-j)} exp(-sum_{k=1}^l E(x_{i+k}, x'_{j+k})/2/sigma_c^2)
        Args:
            seq1 (str): subsequence 1
            seq2 (str): subsequence 2
            sigma_c (float): sigma_c parameter

        Returns:
            torch.tensor(float): B_ij
        """
        t_dict = {0: 1}
        for l in range(len(seq1)):
            a1 = seq1[l:l+1] # current AA in seq 1
            a2 = seq2[l:l+1] # current AA in seq 1
            t_dict[l+1] = self.t_ij[self.str2ij[a1], self.str2ij[a2]]
        B_ij = torch.tensor(0)
        for i in range(2, len(t_dict)+1):
            product = 1
            for j in range(i):
                product *= t_dict[j]
            B_ij = B_ij.add(product)
        return B_ij
    
    def get_GS_deprecated(self, psi1, psi2):
        """
        Calculate Generic String Kernel between seq1 and seq2
        GS = sum_{i=0}^{psi1.shape[1]}sum_{j=0}^{psi2.shape[1]} exp(-(i-j)^2/2/sigma_p^2)B_ij


        Args:
            psi1 (torch.Tensor): 24xl Blosum62 representation of sequence 1
            psi2 (torch.Tensor): 24xl Blosum62 representation of sequence 2
            L (torch.Tensor(int)): Maximum length parameter
            sigma_p (torch.Tensor(float)): sigma_p parameter
            sigma_c (torch.Tensor(float)): sigma_c parameter

        Returns:
            torch.tensor(float): GS
        """
        GS = torch.tensor(0.0)
        for i in range(psi1.shape[1]):
            GS_i = torch.tensor(0.0)
            for j in range(psi2.shape[1]):
                GS_i += torch.exp(-((torch.tensor(i-j, dtype=torch.float))**2).div(2).div(self.lengthscale[0, 0]))*self.get_Bij(psi1, psi2, i, j)
            GS += GS_i
        return GS

    def get_GS(self, seq1, seq2):
        """
        Calculate Generic String Kernel between seq1 and seq2
        GS = sum_{i=0}^{psi1.shape[1]}sum_{j=0}^{psi2.shape[1]} exp(-(i-j)^2/2/sigma_p^2)B_ij


        Args:
            seq1 (str): sequence 1
            seq2 (str): sequence 2
            L (torch.Tensor(int)): Maximum length parameter
            sigma_p (torch.Tensor(float)): sigma_p parameter
            sigma_c (torch.Tensor(float)): sigma_c parameter

        Returns:
            torch.tensor(float): GS
        """
        GS = torch.tensor(0.0)
        for i in range(len(seq1)):
            GS_i = torch.tensor(0.0)
            for j in range(len(seq2)):
                l = min(self.L, len(seq1)-i, len(seq2)-j)
                subseq1 = seq1[i:i+l] # create subsequences
                subseq2 = seq2[j:j+l]
                B_ij_key1 = subseq1 + "_" + subseq2 # create keys for B_ij_dict
                B_ij = self.Bij_dict.get(B_ij_key1, None) # see if this B_ij has been cacluated before
                if B_ij is None: # if not
                    B_ij = self.get_Bij(subseq1, subseq2) # calculate it
                    B_ij_key2 = subseq2 + "_" + subseq1 
                    self.Bij_dict[B_ij_key1] = B_ij # store it in both possible key combinations
                    self.Bij_dict[B_ij_key2] = B_ij
                GS_i += torch.exp(-torch.pow(torch.tensor(i-j), 2.0).div(2.0).div(self.lengthscale[0, 0]))*B_ij
            GS += GS_i
        return GS
    
    def get_gram_matrix_deprecated(self, X, Y, diag=False):
        """
        Calcuate the gram matrix for Generic String Kernel

        Args:
            X (torch.Tensor): N_Xxdxl matrix of sequence Blosum62 representations
            Y (torch.Tensor): N_Yxdxl matrix of sequence Blosum62 representations
            diag (bool): whether to only calculate the diagonal

        Returns:
            torch.Tensor: N_XxN_Y GS kernel gram matrix or N_X diagonal values
        """
        if diag:
            gram_matrix = torch.zeros((X.shape[0], ))
            for i in range(X.shape[0]):
                gram_matrix[i] = self.get_GS(X[i, ...], Y[i, ...])
        else:
            gram_matrix = torch.zeros((X.shape[0], Y.shape[0]))
            if X.shape[0] == Y.shape[0]:
                for i in range(X.shape[0]):
                    for j in range(i, Y.shape[0]):
                        gram_matrix[i, j] = self.get_GS(X[i, ...], Y[j, ...])
                        gram_matrix[j, i] = gram_matrix[i, j]
            else:
                for i in range(X.shape[0]):
                    for j in range(Y.shape[0]):
                        gram_matrix[i, j] = self.get_GS(X[i, ...], Y[j, ...])
        return gram_matrix
    
    def get_gram_matrix(self, X, Y, diag=False):
        """
        Calcuate the gram matrix for Generic String Kernel

        Args:
            X (list): N_X list of sequences
            Y (list): N_Y list of sequences
            L (torch.Tensor(int)): Maximum length parameter
            sigma_p (torch.Tensor(float)): sigma_p parameter
            sigma_c (torch.Tensor(float)): sigma_c parameter
        
        Returns:
            torch.Tensor: N_XxN_Y GS kernel gram matrix
        """
        if diag:
            gram_matrix = torch.zeros((len(X)))
            for i in range(len(X)):
                gram_matrix[i] = self.get_GS(X[i], Y[i])
        else:
            gram_matrix = torch.zeros((len(X), len(Y)))
            if len(X) == len(Y):
                for i in range(len(X)):
                    for j in range(i, len(Y)):
                        gram_matrix[i, j] = self.get_GS(X[i], Y[j])
                        gram_matrix[j, i] = gram_matrix[i, j]
            else:
                for i in range(len(X)):
                    for j in range(len(Y)):
                        gram_matrix[i, j] = self.get_GS(X[i], Y[j])
        return gram_matrix

    def get_kernel(self, X, Y=None, diag=False):
        if Y is not None:
            kernel = self.get_gram_matrix(X, Y, diag=diag)
            diag_X = self.get_gram_matrix(X, X, diag=True)
            diag_Y = self.get_gram_matrix(Y, Y, diag=True)
            kernel /= torch.outer(torch.sqrt(diag_X), torch.sqrt(diag_Y))
        else:
            kernel = self.get_gram_matrix(X, X, diag=diag)
            if diag:
                diag_X = kernel
            else:
                diag_X = torch.diag(kernel)
            kernel /= torch.outer(torch.sqrt(diag_X), torch.sqrt(diag_X))
        return kernel
    
    def forward(self, X, Y=None, diag=False, **params):
        self.t_ij = self.get_tij() # these are functions of the lengthscale
        self.Bij_dict = {} # need to be calculated everytime the kernel is called
        X = X[:, 0]
        X_decoded = self.translator.decode(X)
        if Y is not None:
            Y = Y[:, 0]
            Y_decoded = self.translator.decode(Y)
        K = self.get_kernel(X_decoded,Y_decoded, diag=diag)
        self.t_ij.detach()
        return K