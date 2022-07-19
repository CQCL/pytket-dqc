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
	# Coarsen a pair, then uncoarsen and check hypergraph was restored
	coarse_hyp.coarsen(3,1)
	coarse_hyp.uncoarsen()
	copy_hyp = coarse_hyp.to_static_hypergraph()
	assert orig_hyp.vertex_list == copy_hyp.vertex_list and orig_hyp.hyperedge_list == copy_hyp.hyperedge_list
	# More coarsening then uncoarsening
	coarse_hyp.coarsen(5,4)
	coarse_hyp.coarsen(3,8)
	coarse_hyp.coarsen(7,9)
	coarse_hyp.coarsen(3,1)
	coarse_hyp.uncoarsen()
	coarse_hyp.uncoarsen()
	coarse_hyp.uncoarsen()
	coarse_hyp.uncoarsen()
	copy_hyp = coarse_hyp.to_static_hypergraph()
	assert orig_hyp.vertex_list == copy_hyp.vertex_list and orig_hyp.hyperedge_list == copy_hyp.hyperedge_list

def test_current_neighbours():
	coarse_hyp = CoarseHyp(orig_hyp, [3, 5, 7])
	# Check after one contraction
	coarse_hyp.coarsen(3,1)
	assert {3, 4, 5, 6, 9} == coarse_hyp.current_neighbours(2)
	assert {2, 4, 7, 8, 9} == coarse_hyp.current_neighbours(3)
	assert {2, 5, 7, 9} == coarse_hyp.current_neighbours(6)
	assert {3, 4, 5, 7} == coarse_hyp.current_neighbours(8)
	coarse_hyp.uncoarsen()
	# Check after multiple contractions
	coarse_hyp.coarsen(5,4)
	coarse_hyp.coarsen(3,8)
	assert {1, 5, 6, 9} == coarse_hyp.current_neighbours(2)
	assert {1, 5, 7, 9} == coarse_hyp.current_neighbours(3)
	assert {2, 5, 7, 9} == coarse_hyp.current_neighbours(6)
	coarse_hyp.coarsen(7,9)
	coarse_hyp.coarsen(3,1)
	assert {3, 5, 6, 7} == coarse_hyp.current_neighbours(2)
	assert {2, 5, 7} == coarse_hyp.current_neighbours(3)
	assert {2, 3, 6, 7} == coarse_hyp.current_neighbours(5)
	assert {2, 5, 7} == coarse_hyp.current_neighbours(6)
	assert {2, 3, 5, 6} == coarse_hyp.current_neighbours(7)