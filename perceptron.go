package perceptron

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/nathangreene3/math/linalg/matrix"
	"github.com/nathangreene3/math/linalg/vector"
)

// Decider returns a value on the range [0,1] given some input.
type Decider func(float64) float64

// Perceptron is a set of weights and a bias.
type Perceptron struct {
	dimensions int
	weights    vector.Vector
	bias       float64
	decider    Decider
}

// Stringer does something in using perceptron.String.
var _ = fmt.Stringer(&Perceptron{})

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
	return fmt.Sprintf("[%0.2f], %0.2f", p.weights, p.bias)
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
			log.Fatal("perceptron failed to train")
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

// setWeightsBias force-sets the weights and bias to specified values.
func (p *Perceptron) setWeightsBias(w vector.Vector, b float64) {
	n := len(p.weights)
	if n != len(w) {
		log.Fatal("weight dimension mismatch")
	}

	p.bias = b
	for i := 0; i < n; i++ {
		p.weights[i] = w[i]
	}
}

//-----------------------------------------------------------------------
// Deciders
// These functions return a value on the range [0,1] for any real x. They
// are used to determine what the result of a perceptron should be.
//-----------------------------------------------------------------------

// Threshold is a simple decision function alternative to the logistic
// function (sigmoid) or other decision functions. It returns 1 if x is
// positive and 0 otherwise.
func Threshold(x float64) float64 {
	if 0 < x {
		return 1
	}

	return 0
}

// Sigmoid returns a value on the range (0,1) for any real x.
func Sigmoid(x float64) float64 {
	// Also called the logistic.

	// f(x) = 1/(1 + exp(-x)) = exp(x)/(1 + exp(x)) = 1 - f(-x)
	// See sigmoidDeriv for properties of f'.
	y := math.Exp(x)
	return y / (1 + y)
}

// SigmoidDeriv returns the derivative of the sigmoid function evaluated
// at x.
func SigmoidDeriv(x float64) float64 {
	// f'(x) = f(x)f(-x) = f(x)(1 - f(x))
	// See sigmoid for properties of f.
	y := Sigmoid(x)
	return y * (1 - y)
}
