package main

import (
	"log"
	"math/rand"
)

type neuron struct {
	weights    []float64
	dimensions int
}

func newNeuron(dims int) *neuron {
	if dims < 1 {
		log.Fatal("dims must be positive")
	}

	n := &neuron{
		weights: make([]float64, 0, dims),
	}
	for i := 0; i < dims; i++ {
		n.weights[i] = append(n.weights, 1-2*rand.Float64())
	}
	n.dimensions = dims
	return n
}

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

func (n *neuron) backPropagate(input []float64, target float64) {

}

func (n *neuron) setWeights(weights []float64) {
	if n.dimensions != len(weights) {
		log.Fatal("dimension mismatch")
	}

	copy(n.weights, weights)
}
