package neuralnetwork

import (
	gomath "math"
)

// Activator returns a value on the range [0,1] given some input.
type Activator func(float64) float64

// ReLU returns max{x, 0}.
func ReLU(x float64) float64 {
	return x * Threshold(x)
}

// ReLUDeriv returns the derivative of ReLU which is the threshold of x.
func ReLUDeriv(x float64) float64 {
	return Threshold(x)
}

// Sigmoid returns a value on the range (0,1) for any real x.
func Sigmoid(x float64) float64 {
	// Sigmoid(x) = 1/(1 + exp(-x)) = exp(x)/(1 + exp(x)) = 1 - Sigmoid(-x)
	y := gomath.Exp(x)
	return y / (1 + y)
}

// SigmoidDeriv returns the derivative of the sigmoid function evaluated at x.
func SigmoidDeriv(x float64) float64 {
	// d/dx Sigmoid(x) = Sigmoid(x)(1 - Sigmoid(x))
	y := Sigmoid(x)
	return y * (1 - y)
}

// TanH returns the hyperbolic tangent of x.
func TanH(x float64) float64 {
	return 2*Sigmoid(2*x) - 1
}

// TanHDeriv returns the derivative of the hyperbolic tangent function evaluated at x.
func TanHDeriv(x float64) float64 {
	// d/dx TanH(x) = 4 Sigmoid(2x)(1 - Sigmoid(2x)) = 1 - TanH^2(x)
	y := TanH(x)
	return 1 - y*y
}

// Threshold returns one for x > 0 and zero otherwise.
func Threshold(x float64) float64 {
	if 0 < x {
		return 1
	}

	return 0
}
