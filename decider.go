package neuralnetwork

import (
	"math"
)

// Decider returns a value on the range [0,1] given some input.
type Decider func(float64) float64

// Threshold returns one for x > 0 and zero otherwise.
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

// SigmoidDeriv returns the derivative of the sigmoid function evaluated at x.
func SigmoidDeriv(x float64) float64 {
	// f'(x) = f(x)f(-x) = f(x)(1 - f(x))
	// See sigmoid for properties of f.
	y := Sigmoid(x)
	return y * (1 - y)
}

// TanH returns the hyperbolic tangent of x.
func TanH(x float64) float64 {
	return 2*Sigmoid(2*x) - 1
}

// TanHDeriv returns the derivative of the hyperbolic tangent function evaluated at x.
func TanHDeriv(x float64) float64 {
	y := TanH(x) + 1
	return -y * y / 2
}

// ReLU returns max{x, 0}.
func ReLU(x float64) float64 {
	return x * Threshold(x)
}

// ReLUDeriv returns the derivative of ReLU which is the threshold of x.
func ReLUDeriv(x float64) float64 {
	return Threshold(x)
}
