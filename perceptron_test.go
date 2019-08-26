package neuralnetwork

import (
	"testing"

	"github.com/nathangreene3/math/linalg/matrix"
	"github.com/nathangreene3/math/linalg/vector"
)

func and(x, y float64) float64 {
	if x == 1 {
		return y
	}

	return 0
}

func or(x, y float64) float64 {
	if x == 1 {
		return 1
	}

	return y
}

func xor(x, y float64) float64 {
	if x == 1 {
		if y == 0 {
			return 1
		}
		return 0
	}

	return y
}

func nand(x, y float64) float64 {
	if x == 0 {
		if y == 0 {
			return 1
		}
	}

	return 0
}

func TestPerceptronOnLogicGates(t *testing.T) {
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
		p         = NewPerceptron(2, Threshold)
		x, y      float64
	)

	for i, v := range data {
		x, y = v[0], v[1]
		classAND[i] = and(x, y)
		classNAND[i] = nand(x, y)
		classOR[i] = or(x, y)
		classXOR[i] = xor(x, y)
	}

	p.Train(data, classAND, 0.05, 0.95)
	correct := 100 * p.Verify(data, classAND)
	if correct < 100 {
		t.Fatalf("Class AND result: %0.2f%%", correct)
	}

	p.Train(data, classNAND, 0.05, 0.95)
	correct = 100 * p.Verify(data, classNAND)
	if correct < 100 {
		t.Fatalf("Class NAND result: %0.2f%%", correct)
	}

	p.Train(data, classOR, 0.05, 0.95)
	correct = 100 * p.Verify(data, classOR)
	if correct < 100 {
		t.Fatalf("Class OR result: %0.2f%%", correct)
	}

	// Training on XOR should fail as XOR is not linearly separable.
	// p.Train(data, classXOR, 0.05, 0.95)
	// correct = 100 * p.Verify(data, classXOR)
	// if correct < 100 {
	// 	t.Fatalf("Class XOR result: %0.2f%%", correct)
	// }
}

