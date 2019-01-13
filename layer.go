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
func (lr layer) backPropagate(input []float64, delta, rate float64) {
	for i := range lr {
		lr[i].backPropagate(input, delta, rate)
	}
}

// learn TODO: this is not called by nn.learn. Maybe it should be?
func (lr layer) learn(inputs [][]float64, class []float64, rate float64) {
	for i := range inputs {
		for j := range lr {
			lr.backPropagate(inputs[i], class[i]-lr[j].feedForward(inputs[i], sigmoid), rate)
		}
	}
}
