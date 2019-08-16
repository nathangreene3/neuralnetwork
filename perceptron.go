package perceptron

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/nathangreene3/math/linalg/matrix"
	"github.com/nathangreene3/math/linalg/vector"
)

// Perceptron is a set of weights and a bias.
type Perceptron struct {
	dimensions int
	weights    vector.Vector
	bias       float64
	decider    Decider
}

// New initiates an untrained perceptron of a specified number of dimensions.
func New(dimensions int, decider Decider) *Perceptron {
	// Initially, -1 < weights, bias < 1
	return &Perceptron{
		dimensions: dimensions,
		weights:    vector.New(dimensions, func(i int) float64 { return 1 - 2*rand.Float64() }),
		bias:       1 - 2*rand.Float64(),
		decider:    decider,
	}
}

// BackPropagate adjusts the weights by delta given for a given input.
func (p *Perceptron) BackPropagate(input vector.Vector, delta float64) {
	deltaInput := input.Copy()
	deltaInput.Multiply(delta)
	p.weights.Add(deltaInput)
	p.bias += delta
}

// FeedForward computes the perceptron decision (result) given an input
// value. A decision function must return a value on the range [0,1].
func (p *Perceptron) FeedForward(input vector.Vector) float64 {
	return p.decider(p.weights.Dot(input) + p.bias)
}

// String returns a formatted string representation of a perceptron.
func (p *Perceptron) String() string {
	return fmt.Sprintf("%0.2f, %0.2f", p.weights, p.bias)
}

// Train trains the perceptron given a set of training data (inputs), a
// function accepting training data (trainer), and the learning rate.
func (p *Perceptron) Train(inputs matrix.Matrix, class vector.Vector, rate, accuracy float64) {
	var (
		maxCount = 1000
		n, _     = inputs.Dimensions()
	)

	for p.Verify(inputs, class) < accuracy {
		for i := 0; i < n; i++ {
			p.BackPropagate(inputs[i], rate*(class[i]-p.FeedForward(inputs[i])))
		}

		maxCount--
		if maxCount == 0 {
			fmt.Println(p.String())
			log.Fatalf("perceptron failed to train on %0.2f", class)
		}
	}
}

// Verify returns the ratio of the number of correct classifications to
// the total number of inputs to classify.
func (p *Perceptron) Verify(inputs matrix.Matrix, class vector.Vector) float64 {
	var (
		correct float64
		n, _    = inputs.Dimensions()
	)

	for i := 0; i < n; i++ {
		if p.FeedForward(inputs[i]) == class[i] {
			correct++
		}
	}

	return correct / float64(n)
}
