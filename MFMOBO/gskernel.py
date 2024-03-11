from itertools import repeat
from functools import partial
import torch
from torch.func import vmap
import gpytorch
from gpytorch.constraints import Positive


class GenericStringKernel(gpytorch.kernels.Kernel):
    def __init__(self, translator, L: int = 5, **kwargs) -> None:
        super().__init__(**kwargs)
        # register the raw parameters
        self.register_parameter(
            name='raw_sigma1', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_constraint("raw_sigma1", Positive())

        self.register_parameter(
            name='raw_sigma2', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_constraint("raw_sigma2", Positive())
        # gpytorch doesn't allow multiple length scales for a single input (or idk how to set it up).
        # So here the code is 'tricked' into thinking the input is 2D instead of the actual 1D.
        # this means that gpytorch will throw an exception if the debug mode is on.
        self.translator = translator
        self.encoded_blosum = self.get_blosum()
        self.E_ij = self.get_Eij()
        self.L = L
        self.t_ij = None # these are functions of the lengthscale
    
    @property
    def sigma1(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_sigma1_constraint.transform(self.raw_sigma1)
    
    @sigma1.setter
    def sigma1(self, value):
        # when setting the parameter, apply the constraint inverse transform
        return self._set_sigma1(value)
    
    def _set_sigma1(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sigma1)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_sigma1=self.raw_sigma1_constraint.inverse_transform(value))

    @property
    def sigma2(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_sigma2_constraint.transform(self.raw_sigma2)
    
    @sigma2.setter
    def sigma2(self, value):
        # when setting the parameter, apply the constraint inverse transform
        return self._set_sigma2(value)
    
    def _set_sigma2(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sigma2)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_sigma2=self.raw_sigma2_constraint.inverse_transform(value))

    def get_blosum(self):
        """
        stars are encoded as "O"

        Returns:
            _type_: _description_
        """
        blusom = torch.zeros(24, 24)
        for key1 in self.translator.psi_dict.keys():
            for j, key2 in enumerate(self.translator.psi_dict.keys()):
                blusom[self.translator.AAmap[key1], self.translator.AAmap[key2]] = self.translator.psi_dict[key1][j]
        return blusom
    
    def get_E(self, psi_a, psi_ap):
        """
        Calculate E(a, a')

        Args:
            psi_a (torch.tensor): property array of AA a
            psi_ap (torch.tensor): property array of AA a'
        """
        return torch.sum(torch.square(psi_a-psi_ap), dim=0)

    def get_Eij(self):
        E_ij = torch.zeros(self.encoded_blosum.size(1), self.encoded_blosum.size(1))
        for i in range(E_ij.shape[0]):
            for j in range(i, E_ij.shape[1]):
                psi_i = self.encoded_blosum[i].div(torch.norm(self.encoded_blosum[i]))
                psi_j = self.encoded_blosum[j].div(torch.norm(self.encoded_blosum[j]))
                E_ij[i, j] = self.get_E(psi_i, psi_j)
                E_ij[j, i] = E_ij[i, j]
        return E_ij
    
    def get_tij(self):
        t_ij = torch.exp(-self.E_ij.div(2).div(self.sigma2))
        return t_ij

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
            t_dict[l+1] = torch.index_select(torch.index_select(self.t_ij, 0, a1), 1, a2)
        B_ij = torch.Tensor([0])
        for i in range(2, len(t_dict)+1):
            product = 1
            for j in range(i):
                product *= t_dict[j]
            B_ij = B_ij.add(product)
        return B_ij.squeeze()
    
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
        seq_len = len(seq1)
        GS = []
        for i in range(seq_len):
            GS_i = []
            for j in range(seq_len):
                l = min(self.L, seq_len-i, seq_len-j)
                subseq1 = seq1[i:i+l] # create subsequences
                subseq2 = seq2[j:j+l]
                B_ij = self.get_Bij(subseq1, subseq2) # calculate B_ij
                dist = torch.Tensor([i-j])
                dist = torch.exp(-torch.pow(dist, 2.0).div(2.0).div(self.sigma1))
                GS_i.append(dist*B_ij)
            GS.append(torch.concat(GS_i, dim=0).sum(dim=0).view(1))
        GS = torch.concat(GS, dim=0).sum(dim=0)
        return GS
    
    def GS_over_X(self, X, Y):
        """
        Helper function to iterator over all X for a single Y.
        """
        return vmap(self.get_GS, in_dims=(0, None))(X, Y)
    
    def GS_over_X_diag(self, X, Y):
        """
        Helper function to iterator over all X for a single Y.
        """
        return vmap(self.get_GS, in_dims=(0, 0))(X, Y)

    def GS_over_Y(self, X, Y):
        """
        Helper function to iterator over X and Y for a single batch.
        """
        return vmap(self.GS_over_X, in_dims=(None, 0))(X, Y)

    def get_gram_matrix_parallel(self, X, Y, diag=False):
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
            gram_matrix = vmap(self.GS_over_X_diag, in_dims=(0, 0))(X, Y)
        else:
            gram_matrix = vmap(self.GS_over_Y, in_dims=(0, 0))(X, Y)
            gram_matrix = gram_matrix.transpose(1, 2)
        return gram_matrix


    def get_kernel(self, X, Y=None, diag=False):
        if Y is not None:
            kernel = self.get_gram_matrix_parallel(X, Y, diag=diag)
            diag_X = self.get_gram_matrix_parallel(X, X, diag=True).view(kernel.size(0), -1)
            diag_Y = self.get_gram_matrix_parallel(Y, Y, diag=True).view(kernel.size(0), -1)
            kernel /= torch.einsum('bi,bj->bij', (torch.sqrt(diag_X), torch.sqrt(diag_Y)))
            #kernel /= torch.outer(torch.sqrt(diag_X), torch.sqrt(diag_Y))
        else:
            kernel = self.get_gram_matrix_parallel(X, X, diag=diag)
            if diag:
                diag_X = kernel
            else:
                diag_X = torch.diag(kernel)
            kernel /= torch.einsum('bi,bj->bij', (torch.sqrt(diag_X), torch.sqrt(diag_X)))
            #kernel /= torch.outer(torch.sqrt(diag_X), torch.sqrt(diag_X))
        return kernel
    
    def forward(self, X, Y=None, diag=False, **params):
        self.t_ij = self.get_tij()  # these are functions of the lengthscale
        if X.dim() <= 1:
            X = X.view(1, 1, -1)
        elif X.dim() == 2:
            X = X.unsqueeze(dim=0)
        if Y is not None: # different X and Y
            if Y.dim() <= 1:
                Y = Y.view(1, 1, -1)
            elif Y.dim() == 2:
                Y = Y.unsqueeze(dim=0)
        X = self.translator.translate_to_ord(X)
        Y = self.translator.translate_to_ord(Y)
        K = self.get_kernel(X, Y, diag=diag)
        self.t_ij.detach()
        if K.size(0) == 1:
            K = K.squeeze(0)
        return K