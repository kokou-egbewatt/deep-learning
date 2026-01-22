# https://www.mathworks.com/help/deeplearning/ug/perceptron-neural-networks.html
class Perceptron:
    def __init__(self, input_size: int, learning_rate: float = 0.01):
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number")
        self.weights = [0.0] * input_size
        self.bias = 0.0
        self.learning_rate = float(learning_rate)

    def activation_function(self, x: float) -> int:
        # Hardlimit activation function -- step function
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        # Input validation
        if not hasattr(inputs, '__iter__'):
            raise TypeError("inputs must be iterable")
        inputs = list(inputs)
        if len(inputs) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} inputs, got {len(inputs)}")
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.activation_function(weighted_sum)

    # https://www.mathworks.com/help/deeplearning/ref/learnp.html
    def train(self, training_inputs, labels, epochs, batch_size=None):
        # Input validation
        if not hasattr(training_inputs, '__iter__') or not hasattr(labels, '__iter__'):
            raise TypeError("training_inputs and labels must be iterable")
        training_inputs = list(training_inputs)
        labels = list(labels)
        if len(training_inputs) != len(labels):
            raise ValueError("training_inputs and labels must have the same length")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        n_samples = len(training_inputs)
        if batch_size is not None:
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("batch_size must be a positive integer")

        # Training loop
        for _ in range(epochs):
            if batch_size is None:
                # Online training (default)
                for inputs, label in zip(training_inputs, labels):
                    prediction = self.predict(inputs)
                    error = label - prediction
                    # Update weights and bias
                    self.weights = [
                        w + self.learning_rate * error * i
                        for w, i in zip(self.weights, inputs)
                    ]
                    self.bias += self.learning_rate * error
            else:
                # Batch training
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    batch_inputs = training_inputs[start:end]
                    batch_labels = labels[start:end]
                    weight_updates = [0.0] * len(self.weights)
                    bias_update = 0.0
                    for inputs, label in zip(batch_inputs, batch_labels):
                        prediction = self.predict(inputs)
                        error = label - prediction
                        for idx in range(len(self.weights)):
                            weight_updates[idx] += self.learning_rate * error * inputs[idx]
                        bias_update += self.learning_rate * error
                    self.weights = [w + wu for w, wu in zip(self.weights, weight_updates)]
                    self.bias += bias_update