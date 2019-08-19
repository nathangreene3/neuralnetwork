package neuralnetwork

import (
	"testing"

	"github.com/nathangreene3/math/linalg/matrix"
	"github.com/nathangreene3/math/linalg/vector"
)

func TestDefineNeuralNetwork(t *testing.T) {
	var (
		data = matrix.Matrix{
			vector.Vector{0, 0}, // class: 0
			vector.Vector{0, 1}, // class: 1
			vector.Vector{1, 0}, // class: 1
			vector.Vector{1, 1}, // class: 0
		}
		nnXOR = defineNeuralNetwork(
			defineLayer(
				defineNeuron(vector.Vector{20, 20}, -30),
				defineNeuron(vector.Vector{20, 20}, -10),
			),
			defineLayer(
				defineNeuron(vector.Vector{-60, 60}, -30),
			),
		)
		classXOR, output vector.Vector
	)

	for _, input := range data {
		classXOR = vector.Vector{xor(input[0], input[1])}
		output = nnXOR.Output(input)
		if !output.Approx(classXOR, 0.01) {
			t.Fatalf(
				"test: XOR\n"+
					"   input: %s\n"+
					"  output: %s\n"+
					"expected: %s", input, output, classXOR)
		}
	}
}

func TestTrainNeuralNetwork(t *testing.T) {
	var (
		data = matrix.Matrix{
			vector.Vector{0, 0},
			vector.Vector{0, 1},
			vector.Vector{1, 0},
			vector.Vector{1, 1},
		}
		n        = len(data)
		classAND = vector.Zero(n)
		// classNAND = vector.Zero(n)
		// classOR   = vector.Zero(n)
		// classXOR  = vector.Zero(n)
		nn   = New(2, 2, 1)
		x, y float64
	)

	for i, v := range data {
		x, y = v[0], v[1]
		classAND[i] = and(x, y)
		// classNAND[i] = nand(x, y)
		// classOR[i] = or(x, y)
		// classXOR[i] = xor(x, y)
	}

	nn.Train(data, classAND)
}
