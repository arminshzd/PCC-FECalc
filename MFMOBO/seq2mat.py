import pickle
from typing import Any, AnyStr

import numpy as np
import torch

class Seq2Mat():
    """
    Class to convert list of sequences to tensor of properties
    """

    def __init__(self, psi_dict_dir: AnyStr) -> None:
        with open(psi_dict_dir, 'rb') as f:
            self.psi_dict = pickle.load(f)
        self.encoder = None
        self.decoder = None
        self.AA_encoder = {A: i+10 for i, A in enumerate(self.psi_dict.keys())}
        
    def seq_to_psi(self, seq: AnyStr) -> torch.Tensor:
            """Encode AA string to blosum62 vectors

            Args:
                str (str): AA sequence string
                psi_dict (dict): AA squence blosum62 matrix
            """
            psi = []
            for a in seq:
                psi.append(torch.div(self.psi_dict[a],torch.norm(self.psi_dict[a])))
            return torch.stack(psi).T
    
    def encode_seq(self, seq: AnyStr) -> int:
        code = []
        for A in seq:
            code.append(str(self.AA_encoder[A]))
        return int("".join(code))              

    def encode(self, seqs:list) -> torch.Tensor:
        """encodes a list of sequences into integers

        Args:
            seqs (list): list of sequences
        
        Returns:
            torch.Tensor: Tensor of integers
        """
        #self.encoder = {seq: self.encode_seq(seq) for i, seq in enumerate(seqs)}
        self.encoder = {seq: i for i, seq in enumerate(seqs)}
        self.decoder = {val: key for key, val in self.encoder.items()}
        self.encoded_seqs = torch.tensor([self.encoder[i] for i in seqs])
        return self.encoded_seqs
    
    def transform(self, X:list) -> torch.Tensor:
        if self.encoder is None:
             raise RuntimeError("Translator has not been fit yet.")
        
        return torch.tensor([self.encoder[i] for i in X])
    
    def decode(self, X:torch.Tensor) -> list:
        """
        Translate property tensor to list of strings

        Args:
            X (torch.Tensor): Nx24xL tensor of properties

        Returns:
            list: (N, ) list of sequences of length L
        """
        if self.encoder is None:
             raise RuntimeError("Translator has not been fit yet.")
    
        return [self.decoder[i.item()] for i in X]
    
    def get_psi(self, seqs:list) -> torch.Tensor:
        """
        Generate property tensor from list of encoded inputs

        Args:
            seqs (list): (N, ) list of encoded sequences

        Returns:
            torch.Tensor: Nx24xL tensor of properties
        """
        tensor_list = []
        for seq in seqs:
             tensor_list.append(self.seq_to_psi(self.decoder[seq.item()]))
        return torch.stack(tensor_list, dim=0)
                