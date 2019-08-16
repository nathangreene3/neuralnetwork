package neuralnetwork

import "github.com/nathangreene3/math/linalg/vector"

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

// TODO
func (nn *NeuralNetwork) feedForward(input vector.Vector) {
}
