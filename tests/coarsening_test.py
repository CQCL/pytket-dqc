from pytket_dqc.circuits import Hypergraph, CoarseHyp

orig_hyp = Hypergraph()
orig_hyp.add_vertices(list(range(0, 10)))
orig_hyp.add_hyperedge([1, 3, 4])
orig_hyp.add_hyperedge([1, 2, 4])
orig_hyp.add_hyperedge([3, 4, 8])
orig_hyp.add_hyperedge([4, 5, 7, 8])
orig_hyp.add_hyperedge([3, 7, 9])
orig_hyp.add_hyperedge([5, 6, 7])
orig_hyp.add_hyperedge([2, 5, 6, 9])

def test_dynhyp_constructor():
	coarse_hyp = CoarseHyp(orig_hyp, [3, 5, 7])
	copy_hyp = coarse_hyp.to_static_hypergraph()
	assert orig_hyp.vertex_list == copy_hyp.vertex_list and orig_hyp.hyperedge_list == copy_hyp.hyperedge_list

def test_coarsen_uncoarsen():
	coarse_hyp = CoarseHyp(orig_hyp, [3, 5, 7])
	# Make a deep copy of the hypergraph structure for later comparison
#	vertex_list_copy = deepcopy(coarse_hyp.vertex_list)
#	hyperedge_dict_copy = deepcopy(coarse_hyp.hyperedge_dict)
#	equal_to_copy = lambda: vertex_list_copy == coarse_hyp.vertex_list and hyperedge_dict_copy == coarse_hyp.hyperedge_dict
	# Coarsen a pair, then uncoarsen and check hypergraph was restored
	coarse_hyp.coarsen(3,1)
	coarse_hyp.uncoarsen()
	copy_hyp = coarse_hyp.to_static_hypergraph()
	assert orig_hyp.vertex_list == copy_hyp.vertex_list and orig_hyp.hyperedge_list == copy_hyp.hyperedge_list
	# More coarsening then uncoarsening
	coarse_hyp.coarsen(2,4)
	coarse_hyp.coarsen(2,5)
	coarse_hyp.coarsen(6,9)
	coarse_hyp.coarsen(2,6)
	coarse_hyp.uncoarsen()
	coarse_hyp.uncoarsen()
	coarse_hyp.uncoarsen()
	coarse_hyp.uncoarsen()
	copy_hyp = coarse_hyp.to_static_hypergraph()
	assert orig_hyp.vertex_list == copy_hyp.vertex_list and orig_hyp.hyperedge_list == copy_hyp.hyperedge_list

def test_current_neighbours():
	assert True