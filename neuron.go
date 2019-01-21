package main

import (
	"fmt"
	"log"
	"math/rand"
	"strings"
)

type neuron struct {
	weights    []float64
	dimensions int // Number of weights
}

var _ = fmt.Stringer(&neuron{})

func (n *neuron) String() string {
	a := make([]string, 0, n.dimensions)
	for i := 0; i < n.dimensions; i++ {
		a = append(a, fmt.Sprintf("%0.2f", n.weights[i]))
	}
	return "[" + strings.Join(a, ",") + "]"
}

// newNeuron returns a neuron of a given dimension with random weights on the range (-1,1).
func newNeuron(dims int) *neuron {
	if dims < 1 {
		log.Fatal("dims must be positive")
	}

	n := &neuron{
		weights:    make([]float64, 0, dims),
		dimensions: dims,
	}
	for i := 0; i < dims; i++ {
		n.weights = append(n.weights, 1-2*rand.Float64())
	}
	return n
}

// output returns the dot product of the input with the neuron weights.
func (n *neuron) output(input []float64) float64 {
	if n.dimensions != len(input) {
		log.Fatal("neuron dimension does not match input dimension")
	}

	var v float64
	for i := range n.weights {
		v += n.weights[i] * input[i]
	}
	return v
}

// backPropagate TODO
func (n *neuron) backPropagate(input []float64, target float64) {
	// output := n.output(input)
	for i := range n.weights {
		n.weights[i] += 0.01 * (target)
	}
}

// setWeights sets the neuron weights to given values.
func (n *neuron) setWeights(weights []float64) {
	if n.dimensions != len(weights) {
		log.Fatal("dimension mismatch")
	}

	copy(n.weights, weights)
}
