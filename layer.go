package main

type neuralNetwork []layer
type layer []*perceptron

func (nn neuralNetwork) feedForward(input []float64) float64 {
	var output []float64
	for i := range nn {
		output = nn[i].feedForward(input)
		input = deepCopy(output)
	}
	return output[maxIndex(output)]
}

// feedForward
func (lr layer) feedForward(input []float64) []float64 {
	output := make([]float64, 0, len(lr))
	for i := range lr {
		output = append(output, lr[i].feedForward(input))
	}
	return output
}

// backPropagate
func (lr layer) backPropagate(input []float64, delta, rate float64) {
	for i := range lr {
		lr[i].backPropagate(input, delta, rate)
	}
}

// learn
func (lr layer) learn(inputs [][]float64, class []float64, rate float64) {

}

// maxIndex returns the index of the largest value in a set x.
func maxIndex(x []float64) int {
	var m int
	for i := range x {
		if x[m] < x[i] {
			m = i
		}
	}
	return m
}

func deepCopy(x []float64) []float64 {
	return append(make([]float64, 0, len(x)), x...)
}
