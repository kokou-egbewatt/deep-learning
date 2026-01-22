import pytest
from perceptron.perceptron import Perceptron

def test_init_valid():
	p = Perceptron(2, learning_rate=0.1)
	assert p.weights == [0.0, 0.0]
	assert p.bias == 0.0
	assert p.learning_rate == 0.1

@pytest.mark.parametrize("input_size", [0, -1, 1.5, 'a'])
def test_init_invalid_input_size(input_size):
	with pytest.raises(ValueError):
		Perceptron(input_size)

@pytest.mark.parametrize("lr", [0, -0.1, 'x'])
def test_init_invalid_learning_rate(lr):
	with pytest.raises(ValueError):
		Perceptron(2, learning_rate=lr)

def test_predict_valid():
	p = Perceptron(2)
	p.weights = [1.0, -1.0]
	p.bias = 0.0
	assert p.predict([2, 1]) == 1
	assert p.predict([0, 2]) == 0

@pytest.mark.parametrize("inputs", [1, None, [1]])
def test_predict_invalid_inputs(inputs):
	p = Perceptron(2)
	with pytest.raises((TypeError, ValueError)):
		p.predict(inputs)

def test_train_online_learns_and_predicts():
	# AND logic gate
	X = [[0,0],[0,1],[1,0],[1,1]]
	y = [0,0,0,1]
	p = Perceptron(2, learning_rate=0.1)
	p.train(X, y, epochs=20)
	preds = [p.predict(x) for x in X]
	assert preds == y

def test_train_batch_learns_and_predicts():
	# OR logic gate
	X = [[0,0],[0,1],[1,0],[1,1]]
	y = [0,1,1,1]
	p = Perceptron(2, learning_rate=0.1)
	p.train(X, y, epochs=20, batch_size=2)
	preds = [p.predict(x) for x in X]
	assert preds == y

def test_train_invalid_inputs():
	p = Perceptron(2)
	with pytest.raises(TypeError):
		p.train(None, [1,0], 10)
	with pytest.raises(TypeError):
		p.train([[1,2]], None, 10)
	with pytest.raises(ValueError):
		p.train([[1,2]], [1,0], 10)
	with pytest.raises(ValueError):
		p.train([[1,2]], [1], 0)
	with pytest.raises(ValueError):
		p.train([[1,2]], [1], 10, batch_size=0)
