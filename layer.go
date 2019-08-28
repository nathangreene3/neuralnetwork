package neuralnetwork

import (
	"github.com/nathangreene3/math/linalg/vector"
)

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

// makeLayer given some pre-defined neurons.
func makeLayer(neurons ...*Neuron) *Layer {
	lr := &Layer{neurons: make([]*Neuron, 0, len(neurons))}
	for _, nr := range neurons {
		lr.append(nr)
	}

	return lr
}

// append a neuron.
func (lr *Layer) append(nr *Neuron) {
	lr.neurons = append(lr.neurons, nr)
	lr.size++
}

// backPropagate ...
func (lr *Layer) backPropagate(input vector.Vector, class vector.Vector) {
	if lr.size != len(class) {
		panic("dimension mismatch")
	}

	for i := 0; i < lr.size; i++ {
		lr.neurons[i].backPropagate(input, class[i])
	}
}

// feedForward returns the result of the layer given some input.
func (lr *Layer) feedForward(input vector.Vector) vector.Vector {
	output := vector.Zero(lr.size)
	for i, nr := range lr.neurons {
		output[i] = nr.feedForward(input)
	}

	return output
}

// Output returns the result of the layer given some input.
func (lr *Layer) Output(input vector.Vector) vector.Vector {
	return lr.feedForward(input)
}
