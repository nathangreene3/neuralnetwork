package main

import (
	"fmt"
	"math"
	"math/rand"
)

type perceptron struct {
	weights []float64
	bias    float64
}

func run() {
	p := newPerceptron(2)
	inputs := make([][]float64, 100)
	for i := range inputs {
		inputs[i] = make([]float64, 2)
		for j := range inputs[i] {
			inputs[i][j] = 2*rand.Float64() - 1.0
		}
	}
	p.learn(inputs, isAbove, 0.1)
	fmt.Print(p.verify())
}

func threshold(x float64) float64 {
	if 0 < x {
		return 1
	}
	return 0
}

func logistic(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func newPerceptron(dimensions int) *perceptron {
	return &perceptron{
		weights: make([]float64, dimensions),
		bias:    0,
	}
}

func (p *perceptron) feedForward(input []float64) float64 {
	result := p.bias
	for i := range input {
		result += input[i] * p.weights[i]
	}
	return result
}

func (p *perceptron) backPropagate(input []float64, delta, rate float64) {
	p.bias += rate * delta
	for i := range input {
		p.weights[i] += rate * delta * input[i]
	}
}

func (p *perceptron) learn(inputs [][]float64, fn func([]float64, func(float64) float64) float64, rate float64) {
	for i := range inputs {
		p.backPropagate(inputs[i], fn(inputs[i], logistic)-p.feedForward(inputs[i]), rate)
	}
}

func (p *perceptron) verify() int {
	correct := 0
	result := 0.0
	point := make([]float64, 2)
	for i := 0; i < 100; i++ {
		for j := range point {
			point[j] = 2.0*rand.Float64() - 1.0
		}
		result = p.feedForward(point)
		if result == isAbove(point, line) {
			correct++
		}
	}
	return correct
}

func isAbove(input []float64, fn func(float64) float64) float64 {
	if fn(input[0]) < input[1] {
		return 1
	}
	return 0
}

func line(x float64) float64 {
	return a*x + b
}

var (
	a = 1.0
	b = 1.0
)
