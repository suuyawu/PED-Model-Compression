import numpy as np


def cal_info(X, Y, k):
	"""
		calculate the F-mutual information between a random vector X
		and a random discrete variable Y

		Input:
		X (:obj:`numpy.ndarray`): matrix of size n x d, n observations of X
		where X is a d-dimensional random vector taking continuous values
		Y (:obj:`numpy.ndarray`): matrix of size n x 1, n observations of Y
		where Y is a 1-dimensional random vector taking discrete values
			0,1,...k-1
		k (:obj:`int`): number of discrete values Y can take

		Output:
		info (:obj:`float`): mutual info between X and Y,
			the larger the more dependence
		err (:obj:`bool`): True, if the output can be used,
			and False otherwise

		Example:
		If the X is a 1000 dimensional input, Y is the 10-classification label
		and if there are 2000 observations, then
		X: float-valued matrix of size 2000 x 1000
		Y: discrete-valued vector of size 2000 x 1
		k: integer 10
		The algo returns near-zero if X and Y are independent, and positive otherwise

		The algorithm uses Gaussian approximation for fast implementation
		for each dimension, the mutu
		F(X1, X2) = (1/s1^2 - 1/s2^2)^2 * s2^2 + (m1/s1^2 - m2/s2^2)^2
	"""

	n, d = X.shape
	class_ind = []
	for i in range(k):
		ind = np.where(Y == i)[0]
		if len(ind) > 0: 
			class_ind.append(ind)

	# some class may be unobserved, redefine k
	k0 = len(class_ind)
	if k0 < 2:
		return 0, False

	# use the largest class as reference for pairwise calculation
	ref = np.argmax(np.array([len(class_ind[i]) for i in range(k0)]))
	X1 = X[class_ind[ref], :]
	mu1 = np.mean(X1, axis=0)
	var1 = np.var(X1, axis=0)
	infos = []
	for j in range(k0):
		if j != ref:
			mu2 = np.mean(X[class_ind[j], :], axis=0)
			var2 = np.var(X[class_ind[j], :], axis=0)
			infotemp = np.mean(var1 * (1/var1 - 1/var2) ** 2 +
				(mu1/var1 - mu2/var2) ** 2)
			if np.isinf(infotemp):
				infotemp = 1e10
			if np.isnan(infotemp):
				infotemp = 1e-10
			# dimension averaged info between X1 and each X2
			infos.append(infotemp)
	info = np.max(np.array(infos))
	return info, True


# test codes
if __name__ == '__main__':

	# generate three-class datasets
	k = 3
	n = 100
	d = 100
	X = np.zeros((n, d))
	Y = np.zeros(n)
	thresh1 = -2
	thresh2 = 2
	for i in range(n):
		X[i, :] = np.random.normal(size=d)
		if np.sum(X[i, :]) < thresh1:
			Y[i] = 0
		elif np.sum(X[i, :]) < thresh2:
			Y[i] = 1
		else:
			Y[i] = 2

	info, err = cal_info(X, Y, k)
	print('\ninfo, err: \n', info, err)
