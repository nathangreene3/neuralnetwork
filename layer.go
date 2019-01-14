package main

import (
	"fmt"
	"strings"
)

// A layer is a list of perceptrons.
type layer []*perceptron

var _ = fmt.Stringer(&layer{})

// String formats a layer as rows of each perceptron's default
// string representation.
func (lr layer) String() string {
	s := make([]string, 0, len(lr))
	for i := range lr {
		s = append(s, lr[i].String())
	}
	return strings.Join(s, "\n")
}

// newLayer returns a layer consisting of a number of nodes each of a
// given dimension.
func newLayer(dims, numNodes int) layer {
	if numNodes < 1 {
		panic("number of nodes in layer must be positive")
	}
	if dims < 1 {
		panic("number of dimensions in a perceptron must be positive")
	}

	lr := make(layer, 0, numNodes)
	for i := 0; i < numNodes; i++ {
		lr = append(lr, newPerceptron(dims))
	}
	return lr
}

// feedForward returns the output of each perceptron in this
// layer.
func (lr layer) feedForward(input []float64) []float64 {
	output := make([]float64, 0, len(lr))
	for i := range lr {
		output = append(output, lr[i].feedForward(input, sigmoid))
	}
	return output
}

// backPropagate updates each perceptron in this layer.
func (lr layer) backPropagate(input []float64, delta float64) {
	for i := range lr {
		lr[i].backPropagate(input, delta)
	}
}
