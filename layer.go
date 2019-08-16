package neuralnetwork

import "github.com/nathangreene3/math/linalg/vector"

// Layer is a list of neurons.
type Layer struct {
	neurons []*Neuron
	size    int
}

// newLayer of neurons.
func newLayer(dimensions, size int) *Layer {
	lr := &Layer{neurons: make([]*Neuron, 0, size)}
	for i := 0; i < size; i++ {
		lr.append(newNeuron(dimensions))
	}

	return lr
}

// append a neuron.
func (lr *Layer) append(nr *Neuron) {
	lr.neurons = append(lr.neurons, nr)
	lr.size++
}

func (lr *Layer) feedForward(input vector.Vector) vector.Vector {
	output := vector.Zero(lr.size)
	for i, nr := range lr.neurons {
		output[i] = nr.feedForward(input)
	}

	return output
}
