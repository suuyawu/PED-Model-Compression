import numpy as np
import dcor

def cal_info(X, Y, k):
	"""
		calculate the max energy distance among random vector Xs that condition
		on k classes, associated with the random discrete variable Y

		Requirments:
		dcor, which is on PyPi and can be installed using pip and conda

		Input:
		X (:obj:`numpy.ndarray`): matrix of size n x d, n observations of X
		where X is a d-dimensional random vector taking continuous values
		Y (:obj:`numpy.ndarray`): matrix of size n x 1, n observations of Y
		where Y is a 1-dimensional random vector taking discrete values
			0,1,...k-1
		k (:obj:`int`): number of discrete values Y can take

		Output:
		dist (:obj:`float`): mean energy distance of k assumed conditioanl 
		distributions among Xs, the larger the more difference and the more dependence of Y
		err (:obj:`bool`): True, if the output can be used,
			and False otherwise

		Example:
		If the X is a 1000 dimensional input, Y is the 10-classification label
		and if there are 2000 observations, then
		X: float-valued matrix of size 2000 x 1000
		Y: discrete-valued vector of size 2000 x 1
		k: integer 10
		The algo returns near-zero if X and Y are independent, and positive otherwise
	"""

	n, d = X.shape
	class_ind = []
	for i in range(k):
		ind = np.where(Y == i)[0]
		if len(ind) > 0: 
			class_ind.append(ind)

	# some class may be unobserved, redefine k
	##how to redefine?
	k0 = len(class_ind)
	if k0 < 2:
		return 0, False

	# use the largest class as reference for pairwise calculation
	ref = np.argmax(np.array([len(class_ind[i]) for i in range(k0)]))
	X1 = X[class_ind[ref], :]
	dists = []
	for j in range(k0):
		if j != ref:
			disttemp = dcor.energy_distance(X1, X[class_ind[j], :])
			dists.append(disttemp)
	dist = np.max(np.array(dists))
	return dist, True

# test codes
def test(n1, n2, mu):
	k = 2
	n = n1+n2
	d = 1
	X = np.zeros((n, d))
	Y = np.zeros(n)

	X[0:n1, :] = np.random.normal(loc=0.0, scale=1.0, size=d)
	X[n1:n, :] = np.random.normal(loc=mu, scale=1.0, size=d)
	Y[0:n1] = [0] * n1
	Y[n1:n] = [1] * n2

	dist, err = cal_info(X, Y, k)
	return dist

def run(m, mu):
	dists = []
	for i in range(m):
		dists.append(test(100, 100, mu))
	print(np.mean(dists), np.max(dists), np.min(dists))


if __name__ == '__main__':
	run(100, 0.0)
	run(100, 1.0)
	run(100, 2.0)
	run(100, 5.0)
	run(100, 10.0)
