from numpy import exp, array, random, dot


class neuro:
    def __init__(self):
        random.seed(1)

        self.weights = 2 * random.random((3, 1)) - 1

    def sig(self, x):
        return 1 / (1 + exp(-x))

    def train(self, inputs, outputs, num):
        for iteration in range(num):
            output = self.think(inputs)
            error = outputs - output
            adjustment = dot(inputs.T, error * output * (1 - output))
            self.weights += adjustment

    def think(self, inputs):
        results = self.sig(dot(inputs, self.weights))
        return results


n_ = neuro()
inputs = array([[1, 1, 1], [1, 0, 1], [0, 1, 1]])
outputs = array([[1, 1, 0]]).T

n_.train(inputs, outputs, 10000)

print(n_.think(array([1, 0, 1])))
