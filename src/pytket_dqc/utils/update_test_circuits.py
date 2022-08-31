import os
import pickle


def _update_test_circuits(
    networks, original_circuits, packed_circuits, test_directory
):
    """NOT FOR PRODUCTION
    A lazy way to update the test circuits used in circuits_test.py.
    """
    n_networks = len(networks)
    assert n_networks == len(original_circuits), \
        "There are an unequal number of networks and original circuits."
    assert n_networks == len(packed_circuits), \
        "There are an unequal number of networks and packed circuits."
    _clear_old_test_files(test_directory)

    for i in range(n_networks):
        with open(
            f"{test_directory}/networks/network{i}.pickle",
                "wb") as f:
            pickle.dump(networks[i], f, pickle.HIGHEST_PROTOCOL)
        with open(
            f"{test_directory}/original_circuits/original_circuit{i}.pickle",
                "wb") as f:
            pickle.dump(original_circuits[i], f, pickle.HIGHEST_PROTOCOL)
        with open(
            f"{test_directory}/packed_circuits/packed_circuit{i}.pickle",
                "wb") as f:
            pickle.dump(packed_circuits[i], f, pickle.HIGHEST_PROTOCOL)


def _clear_old_test_files(test_directory):
    for root, _, files in os.walk(test_directory):
        for filename in files:
            if filename.endswith(".pickle"):
                os.remove(os.path.join(root, filename))
