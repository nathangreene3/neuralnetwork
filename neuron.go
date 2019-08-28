package neuralnetwork

import (
	"math/rand"

	"github.com/nathangreene3/math"
	"github.com/nathangreene3/math/linalg/vector"
)

// Neuron is a type of perceptron.
type Neuron struct {
	dimensions int
	weights    vector.Vector
	bias       float64
}

// newNeuron of a given number of dimensions.
func newNeuron(dimensions int) *Neuron {
	return &Neuron{
		dimensions: dimensions,
		weights:    vector.New(dimensions, func(i int) float64 { return 1 - 2*rand.Float64() }),
		bias:       1 - 2*rand.Float64(),
	}

	/*
		return &Neuron{
			p:          NewPerceptron(dimensions, Sigmoid),
			dimensions: dimensions,
		}
	*/
}

// makeNeuron returns a neutron defined by a given set of weights bias.
func makeNeuron(weights vector.Vector, bias float64) *Neuron {
	return &Neuron{
		dimensions: len(weights),
		weights:    weights,
		bias:       bias,
	}

	/*
		return &Neuron{
			p:          DefinePerceptron(weights, bias, Sigmoid),
			dimensions: len(weights),
		}
	*/
}

// backPropagate ...
func (nr *Neuron) backPropagate(input vector.Vector, class float64) {
	var (
		output = nr.feedForward(input)
		// delta  = SigmoidDeriv(output) * (class - output)
		delta      = TanHDeriv(output) * (class - output)
		inputDelta = input.Copy()
	)

	inputDelta.Multiply(delta)
	nr.weights.Add(inputDelta)
	nr.bias += delta
}

// feedForward ...
func (nr *Neuron) feedForward(input vector.Vector) float64 {
	// return Sigmoid(nr.weights.Dot(input) + nr.bias)
	return TanH(nr.weights.Dot(input) + nr.bias)
}

// Output ...
func (nr *Neuron) Output(input vector.Vector) float64 {
	return nr.feedForward(input)
}

// Train ...
func (nr *Neuron) Train(inputs []vector.Vector, classes []float64, accuracy float64) {
	n := len(inputs)
	if n != len(classes) {
		panic("dimension mismatch")
	}

	for maxIters := 1 << 20; nr.verify(inputs, classes) < accuracy && 0 < maxIters; maxIters-- {
		for i := 0; i < n; i++ {
			nr.backPropagate(inputs[i], classes[i])
		}
	}
}

func (nr *Neuron) verify(inputs []vector.Vector, classes []float64) float64 {
	n := len(inputs)
	if n != len(classes) {
		panic("dimension mismatch")
	}

	var correct float64
	for i := 0; i < n; i++ {
		if math.Approx(nr.feedForward(inputs[i]), classes[i], 0.1) {
			correct++
		}
	}

	return correct / float64(n)
}
