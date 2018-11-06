package main

import (
	"fmt"
	"log"
	"math/rand"
)

func main() {
	f, err := getFlowers("iris.csv")
	if err != nil {
		log.Fatalf("%v\n", err)
	}
	fmt.Println(f)
}

// run initiates and trains a new perceptron given a training
// function and data to train with and process.
func run(dimensions int, trainer func([]float64) float64) {
	p := newPerceptron(dimensions)
	inputs := make([][]float64, 100)
	for i := range inputs {
		inputs[i] = make([]float64, dimensions)
		for j := range inputs[i] {
			inputs[i][j] = 2*rand.Float64() - 1.0
		}
	}
	p.learn(inputs, trainer, 0.1)
}
