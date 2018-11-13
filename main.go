package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

func main() {
	runANDGate()
	runORGate()
	runNANDGate()
	runXORGate()
}

func runFlowers() {
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

func runANDGate() {
	n := 100
	data := getBinaryPairs(n)
	class := make([]float64, n)
	for i := range class {
		class[i] = and(data[i])
	}
	fmt.Println(
		run(
			data,
			class,
			func(x []float64, y float64) float64 {
				return y
			},
		),
	)
}

func runORGate() {
	n := 100
	data := getBinaryPairs(n)
	class := make([]float64, n)
	for i := range class {
		class[i] = or(data[i])
	}
	fmt.Println(
		run(
			data,
			class,
			func(x []float64, y float64) float64 {
				return y
			},
		),
	)
}

func runNANDGate() {
	n := 100
	data := getBinaryPairs(n)
	class := make([]float64, n)
	for i := range class {
		class[i] = nand(data[i])
	}
	fmt.Println(
		run(
			data,
			class,
			func(x []float64, y float64) float64 {
				return y
			},
		),
	)
}

// runXORGate demonstrates the failure to classify xor correctly.
func runXORGate() {
	n := 100
	data := getBinaryPairs(n)
	class := make([]float64, n)
	for i := range class {
		class[i] = xor(data[i])
	}
	fmt.Println(
		run(
			data,
			class,
			func(x []float64, y float64) float64 {
				return y
			},
		),
	)
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

// getBinaryPairs returns n random assortments of the binary pairs:
// [0,0], [0,1], [1,0], and [1,1].
func getBinaryPairs(n int) [][]float64 {
	rand.Seed(int64(time.Now().Second()))
	pairs := make([][]float64, n)
	for i := range pairs {
		pairs[i] = make([]float64, 2)
		pairs[i][0] = float64(rand.Intn(2))
		pairs[i][1] = float64(rand.Intn(2))
	}
	return pairs
}

// run trains a new perceptron and returns verification of its successful classification.
func run(data [][]float64, class []float64, trainer func([]float64, float64) float64) float64 {
	count := int(0.75 * float64(len(data)))
	p := newPerceptron(len(data[0]))
	p.learn(data[:count], class[:count], trainer, 0.01)
	return p.verify(data[count:], class[count:])
}
