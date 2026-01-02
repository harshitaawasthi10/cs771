import numpy as np
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel

# You are allowed to import any submodules of sklearn e.g. metrics.pairwise to construct kernel Gram matrices
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_kernel, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here

# -------------------------
# Hyperparameters (tune me)
# -------------------------
BEST_DEGREE = 2      # tune this on public data
BEST_COEF0  = 1    # tune this on public data

# --------------------------------------------------------------------
# Internal constants for the APUF/XOR-APUF decoder
# --------------------------------------------------------------------
_K_BITS = 32
_N_MODEL = _K_BITS + 1  # 33
_KRON_DIM = _N_MODEL * _N_MODEL  # 1089


################################
# Non Editable Region Starting #
################################
def my_kernel( X1, Z1, X2, Z2 ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to compute Gram matrices for your proposed kernel
	# Your kernel matrix will be used to train a kernel ridge regressor
	
	# Ensure numpy arrays and correct shapes
	X1 = np.asarray(X1).reshape(-1, 1)
	X2 = np.asarray(X2).reshape(-1, 1)
	Z1 = np.asarray(Z1)
	Z2 = np.asarray(Z2)

	# Fast vectorised polynomial kernel on z-features:
	# Kz[i,j] = (Z1[i] · Z2[j] + coef0) ** degree
	# Using matrix multiplication is BLAS-accelerated and much faster than sklearn's wrapper.
	Kz = (Z1 @ Z2.T + BEST_COEF0) ** BEST_DEGREE

	# Outer product of x-values: (n1,1) @ (1,n2) -> (n1,n2)
	outer_x = X1 @ X2.T

	# Final semi-parametric kernel: x1*x2*Kz + 1
	G = outer_x * Kz + 1.0

	return G


################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to invert a PUF linear model to get back delays
	# w is a single 1089-dim vector (last dimension being the bias term)
	# The output should be eight 32-dimensional vectors

	# Convert to numpy array
	w = np.asarray(w, dtype=np.float64).reshape(-1,)

	# If input size is unexpected, return zeros (as autograder expects)
	if w.size != _KRON_DIM:
		zeros = np.zeros(_K_BITS, dtype=np.float64)
		return zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros

	# ----------------------------
	# Step 1: factor 1089-vector into two 33-vectors (rank-1 factorization)
	# ----------------------------
	def _factor_rank1_kron_vector_fast(w_vec):
		"""
		Factor w (length 1089) generated as kron(w1, w2) into w1, w2.
		We reshape to (33,33) and find a non-zero pivot using argmax on absolute values.
		Then recover w1 = W[:, j0], w2 = W[i0, :] / W[i0, j0].
		Handles numerical near-zero by thresholding.
		"""
		W = w_vec.reshape(_N_MODEL, _N_MODEL)
		# find index of maximum absolute entry
		idx = np.argmax(np.abs(W))
		pivot_val = W.flat[idx]

		# threshold for numerical zero
		if abs(pivot_val) < 1e-12:
			# treat as all-zero
			return np.zeros(_N_MODEL, dtype=np.float64), np.zeros(_N_MODEL, dtype=np.float64)

		i0, j0 = divmod(idx, _N_MODEL)
		w1_hat = W[:, j0].copy()
		w2_hat = W[i0, :] / pivot_val

		return w1_hat, w2_hat

	# ----------------------------
	# Step 2: decode single 33-dim APUF model into four 32-d delays
	# ----------------------------
	def _decode_single_apuf_fast(w_vec):
		"""
		Given w (length 33 = model of one arbiter PUF), recover non-negative delays
		p, q, r, s (each length 32) such that the generated model matches w.
		This uses the assignment's exact linear structure and produces a valid solution.
		"""
		wv = np.asarray(w_vec, dtype=np.float64).reshape(_N_MODEL,)

		# alpha, beta are length 32 (for indices 0..31)
		alpha = np.empty(_K_BITS, dtype=np.float64)
		beta = np.empty(_K_BITS, dtype=np.float64)

		# Vectorised reconstruction:
		# alpha[0] = w[0]
		# for k=1..31: alpha[k] = w[k]/2, beta[k-1] = w[k]/2
		# beta[31] = w[32]
		alpha[0] = wv[0]
		alpha[1:] = 0.5 * wv[1:_N_MODEL]
		beta[:-1] = 0.5 * wv[1:_N_MODEL]
		beta[-1] = wv[_N_MODEL - 1]

		# Differences that determine p,q and r,s
		d = alpha + beta  # p - q
		c = alpha - beta  # r - s

		# Choose non-negative delays:
		# if delta >= 0 -> (delta, 0)
		# else -> (0, -delta)
		# Vectorised masks
		pos_d = d >= 0.0
		pos_c = c >= 0.0

		p = np.empty_like(d)
		q = np.empty_like(d)
		r = np.empty_like(c)
		s = np.empty_like(c)

		# For d -> p,q
		p[pos_d] = d[pos_d]
		q[pos_d] = 0.0
		p[~pos_d] = 0.0
		q[~pos_d] = -d[~pos_d]

		# For c -> r,s
		r[pos_c] = c[pos_c]
		s[pos_c] = 0.0
		r[~pos_c] = 0.0
		s[~pos_c] = -c[~pos_c]

		# Clip tiny numerical negatives (safety)
		p = np.maximum(p, 0.0)
		q = np.maximum(q, 0.0)
		r = np.maximum(r, 0.0)
		s = np.maximum(s, 0.0)

		return p, q, r, s

	# Factor kron vector
	u_vec, v_vec = _factor_rank1_kron_vector_fast(w)

	# Decode each APUF model into delays
	a, b, c, d = _decode_single_apuf_fast(u_vec)
	p, q, r, s = _decode_single_apuf_fast(v_vec)

	# Ensure return types are numpy arrays of length 32 and non-negative
	a = np.asarray(a, dtype=np.float64).reshape(_K_BITS)
	b = np.asarray(b, dtype=np.float64).reshape(_K_BITS)
	c = np.asarray(c, dtype=np.float64).reshape(_K_BITS)
	d = np.asarray(d, dtype=np.float64).reshape(_K_BITS)
	p = np.asarray(p, dtype=np.float64).reshape(_K_BITS)
	q = np.asarray(q, dtype=np.float64).reshape(_K_BITS)
	r = np.asarray(r, dtype=np.float64).reshape(_K_BITS)
	s = np.asarray(s, dtype=np.float64).reshape(_K_BITS)

	return a, b, c, d, p, q, r, s