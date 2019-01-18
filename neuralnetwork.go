package main

import (
	"fmt"
	"math"
	"strings"
)

// neuralNetwork is an ordered list of layers.
type neuralNetwork []layer

var _ = fmt.Stringer(&neuralNetwork{})

// String formats a neural network as rows of each layer's default string
// representation.
func (nn neuralNetwork) String() string {
	s := make([]string, 0, len(nn))
	for i := range nn {
		s = append(s, nn[i].String()+"\n")
	}
	return strings.Join(s, "\n")
}

// newNeuralNetwork returns a neural network. Each ith layer has a
// specified number of perceptrons all of the same ith dimension. The
// zeroth layer should have perceptron dimesions matching the input.
func newNeuralNetwork(dims int, numNodesPerLayer []int) neuralNetwork {
	n := len(numNodesPerLayer) // Number of layers
	nn := make(neuralNetwork, 0, n)
	nn = append(nn, newLayer(dims, numNodesPerLayer[0])) // Input layer has dimensions equal to input
	for i := 1; i < n; i++ {
		nn = append(nn, newLayer(numNodesPerLayer[i-1], numNodesPerLayer[i]))
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
func (nn neuralNetwork) backPropagate(input []float64, class float64) {
	n := len(nn) // Number of layers in neural network

	// Get all outputs (same as feedForward, but keeping each result)
	outputs := make([][]float64, 0, n)
	output := deepCopy(input)
	outputs = append(outputs, output)
	for i := range nn {
		output = nn[i].feedForward(output)
		outputs = append(outputs, output)
	}

	// Go through all outputs from last to first and alter weights and biases according to the standard rule
	for i := n - 1; 0 < i; i-- {
		for j := range nn[i] {
			nn[i][j].backPropagate(outputs[i-1], outputs[i][j]*(1-outputs[i][j])*(outputs[i][j]-class)) // sigmoidDeriv isn't necessary; already have output
		}
	}
}

// learn trains a neural network given inputs, classification, and a
// learning rate.
func (nn neuralNetwork) learn(inputs [][]float64, class []float64) {
	if len(inputs) != len(class) {
		panic("number of inputs must equal number of classifications")
	}

	e0, e1 := 0.0, 1.0 // Error returned from verification; nn is as good as it is going to get when error is constant
	maxCount := 10000  // Safety check
	for 0.0 < math.Abs(e1-e0) {
		// e0 = e1
		for i := range inputs {
			nn.backPropagate(inputs[i], class[i])
		}
		// e1 = nn.verify(inputs, class)
		// for i := range inputs {
		// 	fmt.Println(nn.feedForward(inputs[i]))
		// }

		maxCount--
		if maxCount == 0 {
			// log.Fatal("neural network failed to learn")
			break
		}
	}
}

// verify returns the ratio of the correct number of classifications to
// the number of tested inputs.
func (nn neuralNetwork) verify(inputs [][]float64, class []float64) float64 {
	n := len(inputs)
	if len(class) != n {
		panic("number of inputs must equal number of classifications")
	}

	count := float64(n)
	correct := count
	for i := range inputs {
		if math.Round(nn.feedForward(inputs[i])) != class[i] {
			correct--
		}
	}
	return correct / count
}
