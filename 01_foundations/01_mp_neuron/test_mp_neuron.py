"""MP神经元模型测试。"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MPNeuron

class TestMPNeuron:
    def test_and_gate(self):
        neuron = MPNeuron(n_features=2)
        neuron.set_logic_gate("AND")
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
        predictions = neuron.predict(X)
        assert np.array_equal(predictions, y)
        assert neuron.score(X, y) == 1.0

    def test_or_gate(self):
        neuron = MPNeuron(n_features=2)
        neuron.set_logic_gate("OR")
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
        predictions = neuron.predict(X)
        assert np.array_equal(predictions, y)
        assert neuron.score(X, y) == 1.0

    def test_not_gate(self):
        neuron = MPNeuron(n_features=1)
        neuron.set_logic_gate("NOT")
        X = np.array([[0], [1]])
        y = np.array([1, 0])
        predictions = neuron.predict(X)
        assert np.array_equal(predictions, y)
        assert neuron.score(X, y) == 1.0

    def test_xor_impossible(self):
        neuron = MPNeuron(n_features=2)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        neuron.set_logic_gate("AND")
        accuracy = neuron.score(X, y)
        assert accuracy < 1.0

    def test_uninitialized(self):
        neuron = MPNeuron(n_features=2)
        with pytest.raises(ValueError):
            neuron.predict(np.array([[0, 0]]))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])