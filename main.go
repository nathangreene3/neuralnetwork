package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(int64(time.Now().Second()))

	// USE THRESHOLD INSTEAD FOR THESE
	// runANDGate()
	// runORGate()
	// runNANDGate()
	// runXORGate()
	// runXLessY()
	// runFlowers()
	// runCircle()
	// runXORGateNN()

	// USE SIGMOID FOR THESE
	runXORGateNN2()
}

// run trains a new perceptron and returns verification of its
// successful classification.
func run(data [][]float64, class []float64) float64 {
	p := newPerceptron(len(data[0]))
	p.learn(threshold, data, class, 0.01)
	return p.verify(data, class, threshold)
}

// runCircle attempts to train a perceptron to determine if
// points are inside a circle. Random points and the center are
// generated on the range [0,1) and the radius is a random
// value on the range [0,0.5).
func runCircle() {
	// center := []float64{rand.Float64(), rand.Float64()}
	// radius := rand.Float64() / 2
	center := []float64{0.5, 0.5}
	radius := float64(0.25)
	n := 1000000
	data := randomData(2, n)
	class := make([]float64, n)
	for i := range class {
		if (data[i][0]-center[0])*(data[i][0]-center[0])+(data[i][1]-center[1])*(data[i][1]-center[1]) <= math.Pow(radius, 2) {
			class[i]++
		}
	}
	fmt.Printf("result: %0.2f\n", run(data, class))
}

// runFlowers attempts to train a perceptron to determine
// flower classification on a set of flowers imported from a
// csv file.
func runFlowers() {
	f, err := getFlowers("iris.csv")
	if err != nil {
		log.Fatalf("%v\n", err)
	}

	f = shuffle(f)
	n := f.Len()
	data := make([][]float64, n)
	class := make([]float64, n)
	for i := range f {
		data[i], class[i] = f[i].values, float64(f[i].label)
	}
	fmt.Printf("result: %0.2f\n", run(data, class))
}

// runXLessY trains a perceptron to determine if the first of two values
// is less than the second.
func runXLessY() {
	n := 10000 // This takes a lot of test data to converge for some reason; it is linearly separable
	data := randomData(2, n)
	class := make([]float64, n)
	for i := range class {
		if data[i][0] < data[i][1] {
			class[i]++
		}
	}
	fmt.Printf("result: %0.2f\n", run(data, class))
}

// runANDGate trains a perceptron to determine if two values are equal
// to one.
func runANDGate() {
	n := 100
	data := binaryPairs(n)
	class := make([]float64, n)
	for i := range class {
		class[i] = and(data[i])
	}
	fmt.Printf("result: %0.2f\n", run(data, class))
}

// runORGate trains a perceptron to determine if at least one of two
// values is one.
func runORGate() {
	n := 100
	data := binaryPairs(n)
	class := make([]float64, n)
	for i := range class {
		class[i] = or(data[i])
	}
	fmt.Printf("result: %0.2f\n", run(data, class))
}

// runNANDGate trains a perceptron to determine if two values are equal
// to zero.
func runNANDGate() {
	n := 100
	data := binaryPairs(n)
	class := make([]float64, 0, n)
	for i := range class {
		class = append(class, nand(data[i]))
	}
	fmt.Printf("result: %0.2f\n", run(data, class))
}

// runXORGate demonstrates the failure to classify xor correctly.
func runXORGate() {
	n := 10000
	data := binaryPairs(n)
	class := make([]float64, n)
	for i := range class {
		class[i] = xor(data[i])
	}
	fmt.Printf("result: %0.2f\n", run(data, class))
}

// runXORGateNN trains two perceptrons on and and nand results on
// binary pairs. The xor logic gate is generated as the nand
// result of and and nand results. This is logically equivalent to
// XOR(x,y) = NAND(AND(x,y), NAND(x,y)).
func runXORGateNN() {
	data := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}} // Expected result: {0, 1, 1, 0}

	// Classify data
	andClass := make([]float64, 4)
	nandClass := make([]float64, 4)
	for i := range data {
		andClass[i] = and(data[i])
		nandClass[i] = nand(data[i])
	}

	// Train perceptrons
	andNeuron := newPerceptron(2)
	nandNeuron := newPerceptron(2)
	rate := 0.01
	for andNeuron.verify(data, andClass, threshold) < 1.0 {
		andNeuron.learn(threshold, data, andClass, rate)
	}
	for nandNeuron.verify(data, nandClass, threshold) < 1.0 {
		nandNeuron.learn(threshold, data, nandClass, rate)
	}
	fmt.Printf("andNeuron: %s\nnandNeuron: %s\n", andNeuron.String(), nandNeuron.String())

	for i := range data {
		fmt.Printf(
			"result on %v: %0.0f\n",
			data[i],
			nandNeuron.feedForward(
				[]float64{
					andNeuron.feedForward(data[i], threshold),
					nandNeuron.feedForward(data[i], threshold),
				},
				threshold,
			),
		)
	}
}

func runXORGateNN2() {
	data := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}} // Expected result: {0, 1, 1, 0}
	class := make([]float64, 0, len(data))
	for i := range data {
		class = append(class, xor(data[i]))
	}
	nn := newNeuralNetwork([]int{2, 2}, []int{2, 1})
	for i := 0; i < 10000; i++ {
		nn.learn(data, class, 0.01)
		fmt.Printf("%0.2f\n", nn.verify(data, class))
		for j := range nn {
			fmt.Println(nn[j].String())
		}
	}
}

// and returns one (true) if both entries are one and zero (false)
// otherwise. Assumes x holds two values on the set {0,1}.
func and(x []float64) float64 {
	if x[0] == 1 && x[1] == 1 {
		return 1
	}
	return 0
}

// nand returns one (true) if neither entries are one and zero (false)
// otherwise. Assumes x holds two values on the set {0,1}.
func nand(x []float64) float64 {
	if x[0] == 0 && x[1] == 0 {
		return 1
	}
	return 0
}

// or returns one (true) if at least one entry is one and zero (false)
// otherwise. Assumes x holds two values on the set {0,1}.
func or(x []float64) float64 {
	if x[0] == 1 || x[1] == 1 {
		return 1
	}
	return 0
}

// xor returns one (true) if exactly one entry is one and the other is
// zero. It returns zero (false) otherwise. Assumes x holds two values
// on the set {0,1}.
func xor(x []float64) float64 {
	if (x[0] == 1 && x[1] == 0) || (x[0] == 0 && x[1] == 1) {
		return 1
	}
	return 0
}

// randomData returns random values on the range [0,1).
func randomData(dims, count int) [][]float64 {
	data := make([][]float64, 0, count)
	for i := 0; i < count; i++ {
		data = append(data, make([]float64, 0, dims))
		for j := 0; j < dims; j++ {
			data[i] = append(data[i], rand.Float64())
		}
	}
	return data
}

// binaryPairs returns n random assortments of the binary
// pairs: (0,0), (0,1), (1,0), and (1,1).
func binaryPairs(n int) [][]float64 {
	data := make([][]float64, 0, n)
	for i := 0; i < n; i++ {
		data = append(
			data,
			[]float64{
				math.Round(rand.Float64()),
				math.Round(rand.Float64()),
			},
		)
	}
	return data
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

// deepCopy returns a new slice containing the values in x.
func deepCopy(x []float64) []float64 {
	return append(make([]float64, 0, len(x)), x...)
}
