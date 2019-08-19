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

// define ...
func defineNeuron(weights vector.Vector, bias float64) *Neuron {
	return &Neuron{
		p:          DefinePerceptron(weights, bias, Sigmoid),
		dimensions: len(weights),
	}
}

// backPropagate ...
func (nr *Neuron) backPropagate(input vector.Vector, class float64) {
	output := nr.feedForward(input)
	nr.p.backPropagate(input, SigmoidDeriv(output)*(class-output))
}

// feedForward ...
func (nr *Neuron) feedForward(input vector.Vector) float64 {
	return nr.p.feedForward(input)
}

// Output ...
func (nr *Neuron) Output(input vector.Vector) float64 {
	return nr.feedForward(input)
}
