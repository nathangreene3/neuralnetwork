package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"strings"
)

// perceptron.go
// Nathan Greene
// Fall 2018

// perceptron is a set of weights and a bias.
type perceptron struct {
	// weights is an ordered set of real values
	weights []float64
	// bias is the default weight applied
	bias float64
}

// Stringer does something in using perceptron.String.
var _ = fmt.Stringer(&perceptron{})

// String returns a formatted string representation of a perceptron. A
// perceptron is represented as
// 	[weights], [bias]: [0.0, ..., 0.0], 0.0.
func (p *perceptron) String() string {
	a := make([]string, 0, len(p.weights))
	for i := range p.weights {
		a = append(a, fmt.Sprintf("%0.2f", p.weights[i]))
	}
	return fmt.Sprintf("[%s], %0.2f", strings.Join(a, ", "), p.bias)
}

// newPerceptron initiates an empty perceptron with a specified number of
// dimensions. All weights and the bias are set to zero.
func newPerceptron(dimensions int) *perceptron {
	// Initially, -1 < weights, bias < 1
	p := &perceptron{
		weights: make([]float64, 0, dimensions),
		bias:    1 - 2*rand.Float64(),
	}
	for i := 0; i < dimensions; i++ {
		p.weights = append(p.weights, 1-2*rand.Float64())
	}
	return p
}

// feedForward computes the perceptron decision (result) given an input
// value. A decision function must return a value on the range [0,1].
func (p *perceptron) feedForward(input []float64, decision func(float64) float64) float64 {
	result := p.bias
	for i := range input {
		result += input[i] * p.weights[i]
	}
	return decision(result) // threshold, sigmoid, etc.
}

// backPropagate adjusts the weights by delta given for a given input.
func (p *perceptron) backPropagate(input []float64, delta float64) {
	p.bias += delta
	for i := range input {
		p.weights[i] += delta * input[i]
	}
}

// learn trains the perceptron given a set of training data (inputs), a
// function accepting training data (trainer), and the learning rate.
func (p *perceptron) learn(decision func(float64) float64, inputs [][]float64, class []float64) {
	maxCount := 1000
	for p.verify(inputs, class, decision) < 0.9 {
		for i := range inputs {
			p.backPropagate(inputs[i], 0.01*(class[i]-p.feedForward(inputs[i], decision)))
		}

		maxCount--
		if maxCount == 0 {
			fmt.Println(p.String())
			log.Fatal("perceptron failed to learn")
		}
	}
}

// verify returns the ratio of the number of correct classifications to
// the total number of inputs to classify.
func (p *perceptron) verify(inputs [][]float64, class []float64, decision func(float64) float64) float64 {
	count := float64(len(inputs)) // Number of inputs
	correct := count              // Number of correct results
	for i := range inputs {
		if p.feedForward(inputs[i], decision) != class[i] {
			correct--
		}
	}
	return correct / count
}

// setWeightsBias force-sets the weights and bias to specified values.
func (p *perceptron) setWeightsBias(w []float64, b float64) {
	if len(p.weights) != len(w) {
		log.Fatal("weight dimension mismatch")
	}
	for i := range p.weights {
		p.weights[i] = w[i]
	}
	p.bias = b
}

//-----------------------------------------------------------------------
// Decision functions
// These functions return a value on the range [0,1] for any real x. They
// are used to determine what the result of a perceptron should be.
//-----------------------------------------------------------------------

// threshold is a simple decision function alternative to the logistic
// function (sigmoid) or other decision functions. It returns 1 if x is
// positive and 0 otherwise.
func threshold(x float64) float64 {
	if 0 < x {
		return 1
	}
	return 0
}

// sigmoid returns a value on the range (0,1) for any real x.
func sigmoid(x float64) float64 {
	return 1 / (1 + 1/math.Exp(x))
}

// sigmoidDeriv returns the derivative of the sigmoid function evaluated
// at x.
func sigmoidDeriv(x float64) float64 {
	// f(x) = 1/(1+exp(-x)) --> df(x)/dx = exp(-x)/(1+exp(-x)) = f(x)(1-f(x))
	// So this effectively returns sigmoid(x)*(1-sigmoid(x))
	e := math.Exp(-x)
	return e / (1 + e)
}
