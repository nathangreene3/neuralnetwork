package neuralnetwork

import (
	"fmt"
	"math/rand"

	"github.com/nathangreene3/math/linalg/vector"
)

// Perceptron is linear learning model consisting of a set of weights and a bias with a deciding function to determine if an input value is one of two classifications. Classifications should be either 0.0 (false) or 1.0 (true).
type Perceptron struct {
	dimensions int
	weights    vector.Vector
	bias       float64
	activator    Activator
}

// NewPerceptron initiates an untrained perceptron of a specified number of dimensions.
func NewPerceptron(dimensions int, activator Activator) *Perceptron {
	// Initially, -1 < weights, bias < 1
	return &Perceptron{
		dimensions: dimensions,
		weights:    vector.New(dimensions, func(i int) float64 { return 1 - 2*rand.Float64() }),
		bias:       1 - 2*rand.Float64(),
		activator:    activator,
	}
}

// backPropagate adjusts the weights by delta given for a given input.
func (p *Perceptron) backPropagate(input vector.Vector, delta float64) {
	// delta = rate * f'(weights*input+bias) * (class-f(weights*input+bias))
	// delta is subtracted in Data Science from Scratch (output - class). Here, it is added (class - output).
	// delta is an argument here because:
	// * the activator's derivative may not be defined (Lookin' at you, Threshold)
	// * the learning rate may not be given or necessary
	deltaInput := input.Copy()
	deltaInput.Multiply(delta)
	p.weights.Add(deltaInput)
	p.bias += delta
}

// DefinePerceptron ...
func DefinePerceptron(weights vector.Vector, bias float64, activator Activator) *Perceptron {
	return &Perceptron{
		dimensions: len(weights),
		weights:    weights.Copy(),
		bias:       bias,
		activator:    activator,
	}
}

// feedForward computes the perceptron decision (result) given an input
// value. A decision function must return a value on the range [0,1].
func (p *Perceptron) feedForward(input vector.Vector) float64 {
	return p.activator(p.weights.Dot(input) + p.bias)
}

// Output ...
func (p *Perceptron) Output(input vector.Vector) float64 {
	return p.feedForward(input)
}

// String returns a formatted string representation of a perceptron.
func (p *Perceptron) String() string {
	return fmt.Sprintf("%0.2f, %0.2f", p.weights, p.bias)
}

// Train trains the perceptron given a set of training data (inputs), a
// function accepting training data (trainer), and the learning rate.
func (p *Perceptron) Train(inputs []vector.Vector, class []float64, rate, accuracy float64) {
	n := len(inputs)
	if n != len(class) {
		panic("dimension mismatch")
	}

	for maxIters := 1 << 10; p.Verify(inputs, class) < accuracy && 0 < maxIters; maxIters-- {
		for i := 0; i < n; i++ {
			p.backPropagate(inputs[i], rate*(class[i]-p.feedForward(inputs[i])))
		}
	}
}

// Verify returns the ratio of the number of correct classifications to
// the total number of inputs to classify.
func (p *Perceptron) Verify(inputs []vector.Vector, class []float64) float64 {
	n := len(inputs)
	if n != len(class) {
		panic("dimension mismatch")
	}

	var correct float64
	for i := 0; i < n; i++ {
		if p.feedForward(inputs[i]) == class[i] {
			correct++
		}
	}

	return correct / float64(n)
}
