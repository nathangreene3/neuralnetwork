package neuralnetwork

import (
	"github.com/nathangreene3/math/linalg/vector"
)

// ---------------------------------------------------------------
// RESOURCES
// ---------------------------------------------------------------
// https://medium.com/analytics-vidhya/demystifying-neural-networks-a-mathematical-approach-part-1-4e10bed61400
// ---------------------------------------------------------------

// NeuralNetwork is a list of layers.
type NeuralNetwork struct {
	layers []*Layer
	size   int
}

// New neural network.
func New(dimensions int, layerSizes ...int) *NeuralNetwork {
	n := len(layerSizes)
	nn := &NeuralNetwork{layers: make([]*Layer, 0, n)}
	for i := 0; i < n; i++ {
		nn.append(newLayer(dimensions, layerSizes[i]))
	}

	return nn
}

// append a layer to a neural network.
func (nn *NeuralNetwork) append(lr *Layer) {
	nn.layers = append(nn.layers, lr)
	nn.size++
}

func (nn *NeuralNetwork) backPropagate(input vector.Vector, class vector.Vector) {
	outputs := nn.feedForward(input)
	for i := nn.size - 1; 0 <= i; i-- {
		nn.layers[i].backPropagate(outputs[i], class)
	}
}

// defineNeuralNetwork ...
func defineNeuralNetwork(layers ...*Layer) *NeuralNetwork {
	nn := &NeuralNetwork{layers: make([]*Layer, 0, len(layers))}
	for _, lr := range layers {
		nn.append(lr)
	}

	return nn
}

// feedForward ...
func (nn *NeuralNetwork) feedForward(input vector.Vector) []vector.Vector {
	output := make([]vector.Vector, 0, nn.size)
	for i, lr := range nn.layers {
		output = append(output, lr.feedForward(input))
		input = output[i].Copy()
	}

	return output
}

// Output ...
func (nn *NeuralNetwork) Output(input vector.Vector) vector.Vector {
	return nn.feedForward(input)[nn.size-1]
}

// Train ...TODO
func (nn *NeuralNetwork) Train(inputs []vector.Vector, class [][]float64, accuracy float64) {
	n := len(inputs)
	if n != len(class) {
		panic("dimension mismatch")
	}

	for maxIters := 1 << 10; nn.Verify(inputs, class) < accuracy && 0 < maxIters; maxIters-- {
		for i := 0; i < n; i++ {
			nn.backPropagate(inputs[i], class[i])
		}
	}
}

// Verify ...TODO
func (nn *NeuralNetwork) Verify(inputs []vector.Vector, class [][]float64) float64 {
	return 0
}
