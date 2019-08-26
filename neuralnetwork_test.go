package neuralnetwork

import (
	"testing"

	"github.com/nathangreene3/math/linalg/matrix"
	"github.com/nathangreene3/math/linalg/vector"
)

func TestNeuronOnLogicGates(t *testing.T) {
	var (
		data = matrix.Matrix{
			vector.Vector{0, 0},
			vector.Vector{0, 1},
			vector.Vector{1, 0},
			vector.Vector{1, 1},
		}
		n         = len(data)
		classAND  = vector.Zero(n)
		classNAND = vector.Zero(n)
		classOR   = vector.Zero(n)
		classXOR  = vector.Zero(n)
		nr        = newNeuron(2)
		x, y      float64
	)

	for i, v := range data {
		x, y = v[0], v[1]
		classAND[i] = and(x, y)
		classNAND[i] = nand(x, y)
		classOR[i] = or(x, y)
		classXOR[i] = xor(x, y)
	}

	nr.train(data, classAND, 0.95)
	correct := 100 * nr.verify(data, classAND)
	if correct < 100 {
		t.Fatalf("Class AND result: %0.2f%%", correct)
	}

	nr.train(data, classNAND, 0.95)
	correct = 100 * nr.verify(data, classNAND)
	if correct < 100 {
		t.Fatalf("Class NAND result: %0.2f%%", correct)
	}

	nr.train(data, classOR, 0.95)
	correct = 100 * nr.verify(data, classOR)
	if correct < 100 {
		t.Fatalf("Class OR result: %0.2f%%", correct)
	}

	// Training on XOR should fail as XOR is not linearly separable.
	// nr.train(data, classXOR, 0.05, 0.95)
	// correct = 100 * nr.verify(data, classXOR)
	// if correct < 100 {
	// 	t.Fatalf("Class XOR result: %0.2f%%", correct)
	// }
}

// TestDefineNeuralNetwork is for sigmoid use only. It will fail on other deciders.
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
				makeNeuron(vector.Vector{20, 20}, -30),
				makeNeuron(vector.Vector{20, 20}, -10),
			),
			defineLayer(
				makeNeuron(vector.Vector{-60, 60}, -30),
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

/*
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

	// nn.Train(data, classAND, 0)
}
*/
