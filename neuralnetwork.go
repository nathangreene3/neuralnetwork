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
func New(layerSizes ...int) *NeuralNetwork {
	n := len(layerSizes)
	nn := &NeuralNetwork{layers: make([]*Layer, 0, n)}
	nn.append(newLayer(layerSizes[0], layerSizes[0]))
	for i := 1; i < n; i++ {
		nn.append(newLayer(layerSizes[i-1], layerSizes[i]))
	}

	return nn
}

// makeNeuralNetwork ...
func makeNeuralNetwork(layers ...*Layer) *NeuralNetwork {
	nn := &NeuralNetwork{layers: make([]*Layer, 0, len(layers))}
	for _, lr := range layers {
		nn.append(lr)
	}

	return nn
}

// append a layer to a neural network.
func (nn *NeuralNetwork) append(lr *Layer) {
	nn.layers = append(nn.layers, lr)
	nn.size++
}

// backPropagate ...
func (nn *NeuralNetwork) backPropagate(input vector.Vector, class vector.Vector) {
	if nn.layers[nn.size-1].size != len(class) {
		panic("dimension mismatch")
	}

	outputs := nn.feedForward(input)
	for i := nn.size - 1; 0 <= i; i-- {
		nn.layers[i].backPropagate(outputs[i], class)
	}
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
func (nn *NeuralNetwork) Train(inputs []vector.Vector, classes []vector.Vector, accuracy float64) {
	n := len(inputs)
	if n != len(classes) {
		panic("dimension mismatch")
	}

	for maxIters := 1 << 20; nn.Verify(inputs, classes) < accuracy && 0 < maxIters; maxIters-- {
		for i := 0; i < n; i++ {
			nn.backPropagate(inputs[i], classes[i])
		}
	}
}

// Verify ...TODO
func (nn *NeuralNetwork) Verify(inputs []vector.Vector, class []vector.Vector) float64 {
	return 0
}
