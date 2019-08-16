package neuralnetwork

import "github.com/nathangreene3/math/linalg/vector"

// Neuron is a type of perceptron.
type Neuron struct {
	p          *Perceptron
	dimensions int
}

// newNeuron of a given number of dimensions.
func newNeuron(dimensions int) *Neuron {
	return &Neuron{
		p:          NewPerceptron(dimensions, Sigmoid),
		dimensions: dimensions,
	}
}

func (nr *Neuron) feedForward(input vector.Vector) float64 {
	return nr.p.FeedForward(input)
}
