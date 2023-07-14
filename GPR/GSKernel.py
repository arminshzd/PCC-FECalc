import numpy as np
import numpy.typing as npt
from pathlib import Path
from .GS_cpp import *
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
from sklearn.gaussian_process.kernels import GenericKernelMixin,NormalizedKernelMixin
from sklearn.base import clone
import os

class GSkernel(GenericKernelMixin,NormalizedKernelMixin,Kernel):
	'''
	Generic String Kernel based on Yutao Ma's implementation of
	Giguère, S. et al. BMC Bioinformatics 14, 82 (2013). https://doi.org/10.1186/1471-2105-14-82
	'''
	def __init__(self, amino_acid_property: str | Path, L: int = 2, sigma_p: float = 1.0, sigma_c: float = 1.0, 
	      length_scale_bounds: npt.ArrayLike = (1e-5, 1e5)) -> None:
		self.sigma_p = sigma_p
		self.sigma_c = sigma_c
		self.L = L
		self.amino_acid_property = amino_acid_property
		self.E_matrix_ = self._compute_E_matrix(self.amino_acid_property)
		self.length_scale_bounds = length_scale_bounds

	def _compute_E_matrix(self, AA_file: str | Path) -> np.ndarray:
		# Read the file
		with open(AA_file) as f:
			AA_cnt = f.readlines()

		amino_acids = []
		nb_descriptor = len(AA_cnt[0].split()) - 1
		aa_descriptors = np.zeros((len(AA_cnt), nb_descriptor))
		
		# Read descriptors
		for i in range(len(AA_cnt)):
			s = AA_cnt[i].split()
			aa_descriptors[i] = np.array([float(x) for x in s[1:]])
			amino_acids.append(s[0])

		# If nb_descriptor == 1, then all normalized aa_descriptors will be 1
		if nb_descriptor > 1:
			# Normalize each amino acid feature vector
			for i in range(len(aa_descriptors)):
				aa_descriptors[i] /= np.linalg.norm(aa_descriptors[i])

		E_mat = np.zeros((128,128))
		for i in range(len(amino_acids)):
			for j in range(i):
				aa_i,aa_j = amino_acids[i],amino_acids[j]
				E_mat[ord(aa_i),ord(aa_j)] = np.sum((aa_descriptors[i]-aa_descriptors[j])**2)
				E_mat[ord(aa_j),ord(aa_i)] = E_mat[ord(aa_i),ord(aa_j)]
		return E_mat	

	def is_stationary(self) -> bool:
		return False

	@property
	def hyperparameter_L(self) -> Hyperparameter:
		return Hyperparameter("L","numeric","fixed")

	@property
	def hyperparameter_sigma_p(self) -> Hyperparameter:
		return Hyperparameter("sigma_p","numeric",self.length_scale_bounds) # type: ignore

	@property
	def hyperparameter_sigma_c(self) -> Hyperparameter:
		return Hyperparameter("sigma_c","numeric",self.length_scale_bounds) # type: ignore

	def _normalize_gradient(self, K: np.ndarray ,dK: np.ndarray) -> np.ndarray:
		'''
		Evaluate the gradient of normalized gram matrix
		'''
		v,w = np.sqrt(np.diag(K)), np.diag(dK)
		part1 = np.outer(v,v)*dK
		part2 = K*(np.outer(v,1./v)*w + np.outer(1./v,v)*(w[:,np.newaxis]))
		return (part1 - 0.5*part2)/(np.outer(v,v)**2)	

	def __call__(self, X: npt.ArrayLike, Y: npt.ArrayLike | None = None, eval_gradient: bool = False) -> tuple:
		''' Evaluate the kernel and optionally its gradient

		Parameters
		-----------
		X : List of strings.Left argument of the returned kernel k(X, Y).

		Y : List of string. Right argument of the returned kernel k(X, Y). Default = None.
			If None, compute K(X,X).

		eval_gradient : bool, default=False
						Determines whether the gradient with respect to the kernel
						hyperparameter is determined. Only supported when Y is None
		-----------

		Returns
		-----------
		K : ndarray of shape (len(X), len(Y))

		K_gradient : ndarray of shape (len(X), len(X), n_dims)
			The gradient of the kernel k(X, X) with respect to the
			hyperparameter of the kernel. Only returned when eval_gradient
			is True.
		-----------
		'''		
		if Y is not None:
			if eval_gradient:
				raise ValueError("Gradient can only be evaluated when Y is None.")
			else:
				K,_,_ = compute_gram_matrix(X,Y,self.E_matrix_,self.L,self.sigma_p,self.sigma_c,False) # type: ignore
				diag_X = compute_diagonal(X,self.E_matrix_,self.L,self.sigma_p,self.sigma_c) # type: ignore
				diag_Y = compute_diagonal(Y,self.E_matrix_,self.L,self.sigma_p,self.sigma_c) # type: ignore
				return K/np.outer(np.sqrt(diag_X),np.sqrt(diag_Y))
		else:
			if eval_gradient:
				K,dK_dsp,dK_dsc = compute_gram_matrix(X,X,self.E_matrix_,self.L,self.sigma_p,self.sigma_c,True) # type: ignore
				dK_dsp = self._normalize_gradient(K,dK_dsp)
				dK_dsc = self._normalize_gradient(K,dK_dsc) 
				diag_X = np.sqrt(np.diagonal(K))
				K = K/(np.outer(diag_X,diag_X))
				dK_dL = np.empty((K.shape[0],K.shape[1],0))
				return K, np.dstack((dK_dL,self.sigma_c*dK_dsc[:,:,np.newaxis],self.sigma_p*dK_dsp[:,:,np.newaxis]))
			else:
				K,_,_ = compute_gram_matrix(X,X,self.E_matrix_,self.L,self.sigma_p,self.sigma_c,True) # type: ignore
				diag_X = np.sqrt(np.diagonal(K))
				K = K/(np.outer(diag_X,diag_X))
				return K

	def clone_with_theta(self, theta):
		cloned = clone(self) # type: ignore
		cloned.theta = theta
		return cloned

	def __repr__(self) -> str:
		return "{0}(L={1}, sigma_c={2:.3g},sigma_p={3:.3g})".format(
			self.__class__.__name__, self.L, self.sigma_c, self.sigma_p)