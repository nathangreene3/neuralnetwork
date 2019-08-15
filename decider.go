package perceptron

import (
	"math"
)

// Decider returns a value on the range [0,1] given some input.
type Decider func(float64) float64

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
