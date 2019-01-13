package main

import "math"

type neuralNetwork []layer

// newNeuralNetwork returns a neural network. Each ith layer
// has a specified number of perceptrons all of the same ith
// dimension. The zeroth layer should have perceptron dimesions
// matching the input.
func newNeuralNetwork(dimsPerLayer, numNodesPerLayer []int) neuralNetwork {
	if len(dimsPerLayer) != len(numNodesPerLayer) {
		panic("dimension mismatch")
	}

	nn := make(neuralNetwork, 0, len(numNodesPerLayer))
	for i := range numNodesPerLayer {
		nn = append(nn, make(layer, 0, numNodesPerLayer[i]))
		for j := 0; j < numNodesPerLayer[i]; j++ {
			nn[i] = append(nn[i], newPerceptron(dimsPerLayer[i]))
		}
	}
	return nn
}

// feedForward returns the output of the neural network.
func (nn neuralNetwork) feedForward(input []float64) float64 {
	output := deepCopy(input)
	for i := range nn {
		output = nn[i].feedForward(output)
	}
	return output[maxIndex(output)]
}

// backPropagate updates each layer in the neural network.
func (nn neuralNetwork) backPropagate(input []float64, delta, rate float64) {
	for i := range nn {
		nn[i].backPropagate(input, delta, rate)
	}
}

// learn TODO
func (nn neuralNetwork) learn(inputs [][]float64, class []float64, rate float64) {
	for i := range inputs {
		nn.backPropagate(inputs[i], class[i]-nn.feedForward(inputs[i]), rate)
	}
}

func (nn neuralNetwork) verify(inputs [][]float64, class []float64) float64 {
	count := float64(len(inputs))
	correct := count
	for i := range inputs {
		if math.Round(nn.feedForward(inputs[i])) != class[i] {
			correct--
		}
	}
	return correct / count
}
