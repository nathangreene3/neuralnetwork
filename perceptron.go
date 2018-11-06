package main

// perceptron consists of a set of weights and a bias.
type perceptron struct {
	// weights is an ordered set of real values
	weights []float64
	// bias is the default weight applied
	bias float64
}

// threshold is a simple decision function alternative to the
// sigmoid or other decision functions. It returns 1 if x is
// positive and 0 otherwise.
func threshold(x float64) float64 {
	if 0 < x {
		return 1
	}
	return 0
}

// newPerceptron initiates an empty perceptron with a specified
// number of dimensions. All weights and the bias are set to
// zero.
func newPerceptron(dimensions int) *perceptron {
	return &perceptron{
		weights: make([]float64, dimensions),
		bias:    0,
	}
}

// feedForward computes the perceptron decision (result) given
// an input value.
func (p *perceptron) feedForward(input []float64) float64 {
	result := p.bias
	for i := range input {
		result += input[i] * p.weights[i]
	}
	return result
}

func (p *perceptron) backPropagate(input []float64, delta, rate float64) {
	p.bias += rate * delta
	for i := range input {
		p.weights[i] += rate * delta * input[i]
	}
}

// learn trains the perceptron given a set of training data
// (inputs), a function accepting training data (trainer), and the
// learning rate.
func (p *perceptron) learn(inputs [][]float64, trainer func([]float64) float64, rate float64) {
	for i := range inputs {
		p.backPropagate(inputs[i], trainer(inputs[i])-p.feedForward(inputs[i]), rate)
	}
}
