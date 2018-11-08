package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

func main() {
	// f, err := getFlowers("iris.csv")
	// if err != nil {
	// 	log.Fatalf("%v\n", err)
	// }
	// f = shuffle(f)
	// data := make([][]float64, f.Len())
	// class := make([]float64, f.Len())
	// for i := range f {
	// 	data[i], class[i] = f[i].values, float64(f[i].label)
	// }

	dims := 1
	var n int
	for i := 0; i < 6; i++ {
		n = int(math.Pow10(i + 1))
		data := getRandData(dims, n)
		class := make([]float64, n)
		for i := range data {
			if 0 < fn(data[i]) {
				class[i]++
			}
		}
		fmt.Printf(
			"result = %0.2f\n",
			run(
				data,
				class,
				func(x []float64, y float64) float64 {
					if int(y) == 1 {
						return 1
					}
					return 0
				},
			),
		)
		fmt.Println()
	}
}

func fn(x []float64) float64 {
	var v float64
	v = 1
	for i := range x {
		v *= x[i]
	}
	return v
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

// run trains a new perceptron and returns verification of its successful classification.
func run(data [][]float64, class []float64, trainer func([]float64, float64) float64) float64 {
	count := int(0.333 * float64(len(data)))
	p := newPerceptron(len(data[0]))
	p.learn(data[:count], class[:count], trainer, 0.01)
	fmt.Print("weights = [ ")
	for i := range p.weights {
		fmt.Printf("%0.2f ", p.weights[i])
	}
	fmt.Print("], ")
	fmt.Printf("bias = %0.2f\n", p.bias)
	return p.verify(data[count:], class[count:])
}
