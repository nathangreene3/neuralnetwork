package main

import (
	"fmt"
	"log"
	"math/rand"
	"strings"
)

// A layer is a list of neurons with a bias.
type layer struct {
	neurons    []*neuron // List of neurons
	bias       float64   // Bias neuron
	numNeurons int       // Number of neurons
	dimensions int       // Number of weights per neuron
}

var _ = fmt.Stringer(&layer{})

// String formats a layer as rows of each perceptron's default
// string representation.
func (lr *layer) String() string {
	a := make([]string, 0, lr.numNeurons)
	for i := 0; i < lr.numNeurons; i++ {
		a = append(a, lr.neurons[i].String())
	}
	return strings.Join(a, "\n")
}

// newLayer returns a layer consisting of a number of nodes each of a
// given dimension.
func newLayer(dims, numNeurons int) *layer {
	if numNeurons < 1 {
		panic("number of nodes in layer must be positive")
	}
	if dims < 1 {
		panic("number of dimensions in a perceptron must be positive")
	}

	lr := &layer{
		neurons:    make([]*neuron, 0, numNeurons),
		bias:       1 - 2*rand.Float64(),
		numNeurons: numNeurons,
		dimensions: dims,
	}
	for i := 0; i < numNeurons; i++ {
		lr.neurons = append(lr.neurons, newNeuron(dims))
	}
	return lr
}

// feedForward returns the output of each perceptron in this
// layer.
func (lr *layer) feedForward(input []float64) []float64 {
	if lr.dimensions != len(input) {
		log.Fatal("layer dimension mismatch")
	}

	output := make([]float64, 0, lr.numNeurons)
	for i := range lr.neurons {
		output = append(output, lr.neurons[i].output(input))
	}
	return append(output, lr.bias)
}

// backPropagate updates each perceptron in this layer.
func (lr *layer) backPropagate(input []float64, target float64) {
	if lr.dimensions != len(input) {
		log.Fatal("layer dimension mismatch")
	}

	for i := range lr.neurons {
		lr.neurons[i].backPropagate(input, target)
	}
}

func (lr *layer) setBias(b float64) {
	lr.bias = b
}
