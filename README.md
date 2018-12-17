# GoPerceptron

A perceptron is effectively a single neuron that can make binary decisions in classifying data. This data must be linearly separable to correctly classify data. It can make decisions on multi-dimensional data and can mimic the decision-making process of AND, NAND, and OR gates with ease. It cannot be trained on linearly-inseparable data, that is, n-dimensional data that cannot be separated by an (n-1)-dimensional object. In two dimensions, this is equivalent to being able to separate classes by a single line.

The XOR problem is the inability of a single perceptron to be trained to mimic an XOR gate perfectly.  The XOR problem has been *solved* by using two perceptrons, one trained to identify AND conditions and one trained to identify NAND conditions.  Binary pair data is sent to both perceptrons and their output is sent to the NAND perceptron.  A result of one indicates the input binary pair was either (0,1) or (1,0).
