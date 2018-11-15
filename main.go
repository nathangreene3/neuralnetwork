package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"
)

func main() {
	runANDGate()
	runORGate()
	runNANDGate()
	runXORGate()
	// runFlowers()
	// runCircle()
}

// run trains a new perceptron and returns verification of its
// successful classification.
func run(data [][]float64, class []float64) float64 {
	p := newPerceptron(len(data[0]))
	p.learn(data, class, 0.01)
	return p.verify(data, class)
}

// getRandData returns random values on the range [0,1).
func getRandData(dims, count int) [][]float64 {
	rand.Seed(int64(time.Now().Second()))
	data := make([][]float64, count)
	for i := 0; i < count; i++ {
		data[i] = make([]float64, dims)
		for j := 0; j < dims; j++ {
			data[i][j] = rand.Float64()
		}
	}
	return data
}

// getBinaryPairs returns n random assortments of the binary
// pairs: (0,0), (0,1), (1,0), and (1,1).
func getBinaryPairs(n int) [][]float64 {
	rand.Seed(int64(time.Now().Second()))
	pairs := make([][]float64, n)
	for i := range pairs {
		pairs[i] = []float64{float64(rand.Intn(2)), float64(rand.Intn(2))}
	}
	return pairs
}

// runCircle attempts to train a perceptron to determine if
// points are inside a circle. Random points and the center are
// generated on the range [0,1) and the radius is a random
// value on the range [0,0.5).
func runCircle() {
	rand.Seed(int64(time.Now().Second()))
	center := []float64{rand.Float64(), rand.Float64()}
	radius := rand.Float64() / 2
	n := 1000
	data := getRandData(2, n)
	class := make([]float64, n)
	for i := range class {
		if (data[i][0]-center[0])*(data[i][0]-center[0])+(data[i][1]-center[1])*(data[i][1]-center[1]) <= math.Pow(radius, 2) {
			class[i] = 1
		}
	}
	fmt.Printf("%0.2f\n", run(data, class))
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
	fmt.Printf("result = %0.2f\n", run(data, class))
}

func runXLessY() {
	dims := 10
	var n int
	for i := 0; i < 6; i++ {
		n = int(math.Pow10(i + 1))
		data := getRandData(dims, n)
		class := make([]float64, n)
		for i := range data {
			if fn(data[i]) < data[i][0] {
				class[i]++
			}
		}
		fmt.Printf("result = %0.2f\n", run(data, class))
	}
}

func runANDGate() {
	n := 100
	data := getBinaryPairs(n)
	class := make([]float64, n)
	for i := range class {
		class[i] = and(data[i])
	}
	fmt.Println(run(data, class))
}

func runORGate() {
	n := 100
	data := getBinaryPairs(n)
	class := make([]float64, n)
	for i := range class {
		class[i] = or(data[i])
	}
	fmt.Println(run(data, class))
}

func runNANDGate() {
	n := 100
	data := getBinaryPairs(n)
	class := make([]float64, n)
	for i := range class {
		class[i] = nand(data[i])
	}
	fmt.Println(run(data, class))
}

// runXORGate demonstrates the failure to classify xor correctly.
func runXORGate() {
	n := 100
	data := getBinaryPairs(n)
	class := make([]float64, n)
	for i := range class {
		class[i] = xor(data[i])
	}
	fmt.Println(run(data, class))
}

func and(x []float64) float64 {
	if x[0] == 1 {
		if x[1] == 1 {
			return 1
		}
	}
	return 0
}

func nand(x []float64) float64 {
	if x[0] == 0 {
		if x[1] == 0 {
			return 1
		}
	}
	return 0
}

func or(x []float64) float64 {
	if x[0] == 1 {
		return 1
	}
	if x[1] == 1 {
		return 1
	}
	return 0
}

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

func fn(x []float64) float64 {
	v := float64(0)
	for i := 1; i < len(x); i++ {
		v += x[i]
	}
	return v
}
