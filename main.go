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
	runXORGateNN()

	// USE SIGMOID FOR THESE
	// runXORGateNN2()
}

// run trains a new perceptron and returns verification of its successful
// classification.
func runP(data [][]float64, class []float64) (*perceptron, float64) {
	p := newPerceptron(len(data[0]))
	p.learn(threshold, data, class)
	return p, p.verify(data, class, threshold)
}

// runNN TODO
func runNN(data [][]float64, class []float64) (neuralNetwork, float64) {
	return nil, 0
}

//-----------------------------------------------------------------------
// Training scenarios
// These functions train a perceptron or a neural network to classify
// various types of data.
//-----------------------------------------------------------------------

// runCircle attempts to train a perceptron to determine if points are
// inside a circle. Random points and the center are generated on the
// range [0,0.5). The radius is generated as a random value on the range
// [0,0.5).
func runCircle() {
	center := []float64{rand.Float64(), rand.Float64()}
	radius := rand.Float64() / 2
	n := 1000
	data := randomData(2, n)
	class := make([]float64, n)
	for i := range class {
		if (data[i][0]-center[0])*(data[i][0]-center[0])+(data[i][1]-center[1])*(data[i][1]-center[1]) <= math.Pow(radius, 2) {
			class[i]++
		}
	}
	_, result := runP(data, class)
	fmt.Printf("result: %0.2f\n", result)
}

// runFlowers attempts to train a perceptron to determine flower
// classification on a set of flowers imported from a csv file.
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
	_, result := runP(data, class)
	fmt.Printf("result: %0.2f\n", result)
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
	_, result := runP(data, class)
	fmt.Printf("result: %0.2f\n", result)
}

// runANDGate trains a perceptron to determine if two values are equal to
// one.
func runANDGate() {
	n := 100
	data := binaryPairs(n)
	class := make([]float64, n)
	for i := range class {
		class[i] = and(data[i])
	}
	_, result := runP(data, class)
	fmt.Printf("result: %0.2f\n", result)
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
	_, result := runP(data, class)
	fmt.Printf("result: %0.2f\n", result)
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
	_, result := runP(data, class)
	fmt.Printf("result: %0.2f\n", result)
}

// runXORGate demonstrates the failure to classify xor correctly.
func runXORGate() {
	n := 10000
	data := binaryPairs(n)
	class := make([]float64, n)
	for i := range class {
		class[i] = xor(data[i])
	}
	_, result := runP(data, class)
	fmt.Printf("result: %0.2f\n", result)
}

// runXORGateNN trains two perceptrons on and and nand results on binary
// pairs. The xor logic gate is generated as the nand result of and and
// nand results. This is logically equivalent to
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
	for andNeuron.verify(data, andClass, threshold) < 1.0 {
		andNeuron.learn(threshold, data, andClass)
	}
	for nandNeuron.verify(data, nandClass, threshold) < 1.0 {
		nandNeuron.learn(threshold, data, nandClass)
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

// runXORGateNN2 trains a neural network to solve the xor problem.
func runXORGateNN2() {
	data := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}} // Expected result: {0, 1, 1, 0}
	class := make([]float64, 0, len(data))
	for i := range data {
		class = append(class, xor(data[i]))
	}

	nn := newNeuralNetwork([]int{2, 2}, []int{2, 1})
	for i := 0; i < 100; i++ {
		nn.learn(data, class)
		fmt.Printf("%0.2f\n", nn.verify(data, class))
	}
	fmt.Println(nn.String())
}

//-----------------------------------------------------------------------
// Classification functions
// These functions are helper functions that return the classification of
// an input for training perceptrons and neural networks.
//-----------------------------------------------------------------------

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
	if x[0] == 1 {
		return 1
	}
	if x[1] == 1 {
		return 1
	}
	return 0
}

// xor returns one (true) if exactly one entry is one and the other is
// zero. It returns zero (false) otherwise. Assumes x holds two values
// on the set {0,1}.
func xor(x []float64) float64 {
	if x[0] == 1 {
		if x[1] == 0 {
			return 1
		}
	} else if x[1] == 1 {
		return 1
	}
	return 0
}

//-----------------------------------------------------------------------
// Helper functions
// These functions get and manipulate data for training neural netwoks
// and perceptrons.
//-----------------------------------------------------------------------

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
		data = append(data, []float64{math.Round(rand.Float64()), math.Round(rand.Float64())})
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
