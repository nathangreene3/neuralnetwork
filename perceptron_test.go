package perceptron

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
	}

	return 0
}

func nand(x, y float64) float64 {
	if x == 0 {
		if y == 0 {
			return 1
		}
	}

	return 0
}

func TestPerceptron(t *testing.T) {
	var (
		data = matrix.Matrix{
			vector.Vector{0, 0},
			vector.Vector{0, 1},
			vector.Vector{1, 0},
			vector.Vector{1, 1},
		}
		classAND  = vector.Vector{0, 0, 0, 1}
		classNAND = vector.Vector{1, 0, 0, 0}
		classOR   = vector.Vector{0, 1, 1, 1}
		classXOR  = vector.Vector{0, 1, 1, 0}
		p         = New(2, Threshold)
	)

	p.Train(data, classAND, 0.1, 0.9)
	correct := 100 * p.Verify(data, classAND)
	if correct < 100 {
		t.Fatalf("Class AND result: %0.2f%%", correct)
	}

	p.Train(data, classNAND, 0.1, 0.9)
	correct = 100 * p.Verify(data, classNAND)
	if correct < 100 {
		t.Fatalf("Class NAND result: %0.2f%%", correct)
	}

	p.Train(data, classOR, 0.1, 0.9)
	correct = 100 * p.Verify(data, classOR)
	if correct < 100 {
		t.Fatalf("Class OR result: %0.2f%%", correct)
	}

	p.Train(data, classXOR, 0.1, 0.9)
	correct = 100 * p.Verify(data, classXOR)
	if correct < 100 {
		t.Fatalf("Class XOR result: %0.2f%%", correct)
	}
}
