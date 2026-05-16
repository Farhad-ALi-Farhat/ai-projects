import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def attention(Q, K, V):
    d_k = Q.shape[-1]

    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    weights = softmax(scores)

    output = np.dot(weights, V)

    return output, weights

def main():
    Q = np.array([[1, 0, 1]])
    K = np.array([[1, 0, 1],
                  [0, 1, 0],
                  [1, 1, 0]])
    V = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])

    output, weights = attention(Q, K, V)

    print("Attention Weights:\n", weights)
    print("Output:\n", output)

if __name__ == "__main__":
    main()
