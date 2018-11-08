package main

import (
	"fmt"
	"log"
)

func main() {
	f, err := getFlowers("iris.csv")
	if err != nil {
		log.Fatalf("%v\n", err)
	}
	f = shuffle(f)
	data := make([][]float64, f.Len())
	class := make([]float64, f.Len())
	for i := range f {
		data[i], class[i] = f[i].values, float64(f[i].label)
	}

	// n := 1000
	// data := make([][]float64, n)
	// class := make([]float64, n)
	// for i := 0; i < n; i++ {
	// 	data[i] = []float64{rand.Float64(), rand.Float64()}
	// 	if data[i][0] < data[i][1] {
	// 		class[i]++
	// 	}
	// }

	for i := 0; i < 3; i++ {
		fmt.Printf(
			"%0.2f\n",
			run(
				data,
				class,
				func(x []float64, y float64) float64 {
					if int(y) == i+1 {
						return 1
					}
					return 0
				},
			),
		)
	}
}

func run(data [][]float64, class []float64, trainer func([]float64, float64) float64) float64 {
	count := int(0.75 * float64(len(data)))
	p := newPerceptron(len(data[0]))
	p.learn(data[:count], class[:count], trainer, 0.01)
	return p.verify(data[count:], class[count:])
}
