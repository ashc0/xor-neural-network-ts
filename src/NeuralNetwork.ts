import {
  Matrix,
  zerosMatrix,
  sigmoid,
  sigmoidDerivative,
  tanh,
  tanhDerivative,
  dot,
  transpose,
  heMatrix,
} from './utils'

class NeuralNetwork {
  // 神经网络的输入层、隐藏层1、隐藏层2和输出层的大小
  inputSize: number
  hiddenSize1: number
  hiddenSize2: number
  outputSize: number
  // 学习率
  learningRate: number
  // 权重矩阵和偏置矩阵
  weights1: Matrix
  weights2: Matrix
  weights3: Matrix
  bias1: Matrix
  bias2: Matrix
  bias3: Matrix
  // Adam优化算法的中间变量
  m: { [key: string]: Matrix }
  v: { [key: string]: Matrix }
  beta1: number
  beta2: number
  epsilon: number

  constructor(
    inputSize: number,
    hiddenSize1: number,
    hiddenSize2: number,
    outputSize: number,
    learningRate: number = 0.001
  ) {
    this.inputSize = inputSize
    this.hiddenSize1 = hiddenSize1
    this.hiddenSize2 = hiddenSize2
    this.outputSize = outputSize
    this.learningRate = learningRate

    // 初始化权重矩阵和偏置矩阵
    this.weights1 = heMatrix(this.inputSize, this.hiddenSize1)
    this.weights2 = heMatrix(this.hiddenSize1, this.hiddenSize2)
    this.weights3 = heMatrix(this.hiddenSize2, this.outputSize)
    this.bias1 = zerosMatrix(1, this.hiddenSize1)
    this.bias2 = zerosMatrix(1, this.hiddenSize2)
    this.bias3 = zerosMatrix(1, this.outputSize)

    // 初始化Adam优化算法的中间变量
    this.m = {
      weights1: zerosMatrix(this.inputSize, this.hiddenSize1),
      weights2: zerosMatrix(this.hiddenSize1, this.hiddenSize2),
      weights3: zerosMatrix(this.hiddenSize2, this.outputSize),
      bias1: zerosMatrix(1, this.hiddenSize1),
      bias2: zerosMatrix(1, this.hiddenSize2),
      bias3: zerosMatrix(1, this.outputSize),
    }

    this.v = {
      weights1: zerosMatrix(this.inputSize, this.hiddenSize1),
      weights2: zerosMatrix(this.hiddenSize1, this.hiddenSize2),
      weights3: zerosMatrix(this.hiddenSize2, this.outputSize),
      bias1: zerosMatrix(1, this.hiddenSize1),
      bias2: zerosMatrix(1, this.hiddenSize2),
      bias3: zerosMatrix(1, this.outputSize),
    }

    this.beta1 = 0.9
    this.beta2 = 0.999
    this.epsilon = 1e-8
  }

  /**
   * 前向传播
   * @param input 输入矩阵
   * @returns 隐藏层1、隐藏层2和输出层的矩阵
   */
  feedForward(input: Matrix): [Matrix, Matrix, Matrix] {
    // 计算隐藏层1的输出
    const hidden1 = dot(input, this.weights1).map((row, i) =>
      row.map((val, j) => tanh(val + this.bias1[0][j]))
    )
    // 计算隐藏层2的输出
    const hidden2 = dot(hidden1, this.weights2).map((row, i) =>
      row.map((val, j) => tanh(val + this.bias2[0][j]))
    )
    // 计算输出层的输出
    const output = dot(hidden2, this.weights3).map((row, i) =>
      row.map((val, j) => sigmoid(val + this.bias3[0][j]))
    )
    return [hidden1, hidden2, output]
  }

  /**
   * Adam优化
   * @param param 参数矩阵
   * @param grad 梯度矩阵
   * @param m m矩阵
   * @param v v矩阵
   * @param t 迭代次数
   * @returns 更新后的参数矩阵
   */
  adamUpdate(
    param: Matrix,
    grad: Matrix,
    m: Matrix,
    v: Matrix,
    t: number
  ): Matrix {
    // 计算m_hat
    const m_hat = m.map((row, i) =>
      row.map((val, j) => val / (1 - Math.pow(this.beta1, t)))
    )
    // 计算v_hat
    const v_hat = v.map((row, i) =>
      row.map((val, j) => val / (1 - Math.pow(this.beta2, t)))
    )
    // 更新参数矩阵
    return param.map((row, i) =>
      row.map(
        (val, j) =>
          val +
          (this.learningRate * m_hat[i][j]) /
            (Math.sqrt(v_hat[i][j]) + this.epsilon)
      )
    )
  }

  /**
   * 反向传播
   * @param input 输入矩阵
   * @param hidden1 隐藏层1矩阵
   * @param hidden2 隐藏层2矩阵
   * @param output 输出矩阵
   * @param target 目标矩阵
   * @param t 迭代次数
   */
  backpropagate(
    input: Matrix,
    hidden1: Matrix,
    hidden2: Matrix,
    output: Matrix,
    target: Matrix,
    t: number
  ): void {
    // 计算输出层的误差
    const outputError = target.map((row, i) =>
      row.map((val, j) => val - output[i][j])
    )
    // 计算输出层的delta
    const outputDelta = outputError.map((row, i) =>
      row.map((val, j) => val * sigmoidDerivative(output[i][j]))
    )

    // 计算隐藏层2的误差
    const hidden2Error = dot(outputDelta, transpose(this.weights3))
    // 计算隐藏层2的delta
    const hidden2Delta = hidden2Error.map((row, i) =>
      row.map((val, j) => val * tanhDerivative(hidden2[i][j]))
    )

    // 计算隐藏层1的误差
    const hidden1Error = dot(hidden2Delta, transpose(this.weights2))
    // 计算隐藏层1的delta
    const hidden1Delta = hidden1Error.map((row, i) =>
      row.map((val, j) => val * tanhDerivative(hidden1[i][j]))
    )

    // 计算隐藏层2的转置矩阵
    const hidden2T = transpose(hidden2)
    // 计算隐藏层1的转置矩阵
    const hidden1T = transpose(hidden1)
    // 计算输入的转置矩阵
    const inputT = transpose(input)

    // 计算权重矩阵和偏置矩阵的梯度
    const grad_weights3 = dot(hidden2T, outputDelta)
    const grad_weights2 = dot(hidden1T, hidden2Delta)
    const grad_weights1 = dot(inputT, hidden1Delta)

    const grad_bias3: Matrix = [
      outputDelta.reduce(
        (sum, row) => row.map((val, j) => sum[j] + val),
        Array(this.outputSize).fill(0)
      ),
    ]
    const grad_bias2: Matrix = [
      hidden2Delta.reduce(
        (sum, row) => row.map((val, j) => sum[j] + val),
        Array(this.hiddenSize2).fill(0)
      ),
    ]
    const grad_bias1: Matrix = [
      hidden1Delta.reduce(
        (sum, row) => row.map((val, j) => sum[j] + val),
        Array(this.hiddenSize1).fill(0)
      ),
    ]

    this.m.weights3 = this.m.weights3.map((row, i) =>
      row.map(
        (val, j) => this.beta1 * val + (1 - this.beta1) * grad_weights3[i][j]
      )
    )
    this.m.weights2 = this.m.weights2.map((row, i) =>
      row.map(
        (val, j) => this.beta1 * val + (1 - this.beta1) * grad_weights2[i][j]
      )
    )
    this.m.weights1 = this.m.weights1.map((row, i) =>
      row.map(
        (val, j) => this.beta1 * val + (1 - this.beta1) * grad_weights1[i][j]
      )
    )
    this.m.bias3 = this.m.bias3.map((row, i) =>
      row.map(
        (val, j) => this.beta1 * val + (1 - this.beta1) * grad_bias3[i][j]
      )
    )
    this.m.bias2 = this.m.bias2.map((row, i) =>
      row.map(
        (val, j) => this.beta1 * val + (1 - this.beta1) * grad_bias2[i][j]
      )
    )
    this.m.bias1 = this.m.bias1.map((row, i) =>
      row.map(
        (val, j) => this.beta1 * val + (1 - this.beta1) * grad_bias1[i][j]
      )
    )

    this.v.weights3 = this.v.weights3.map((row, i) =>
      row.map(
        (val, j) =>
          this.beta2 * val + (1 - this.beta2) * Math.pow(grad_weights3[i][j], 2)
      )
    )
    this.v.weights2 = this.v.weights2.map((row, i) =>
      row.map(
        (val, j) =>
          this.beta2 * val + (1 - this.beta2) * Math.pow(grad_weights2[i][j], 2)
      )
    )
    this.v.weights1 = this.v.weights1.map((row, i) =>
      row.map(
        (val, j) =>
          this.beta2 * val + (1 - this.beta2) * Math.pow(grad_weights1[i][j], 2)
      )
    )
    this.v.bias3 = this.v.bias3.map((row, i) =>
      row.map(
        (val, j) =>
          this.beta2 * val + (1 - this.beta2) * Math.pow(grad_bias3[i][j], 2)
      )
    )
    this.v.bias2 = this.v.bias2.map((row, i) =>
      row.map(
        (val, j) =>
          this.beta2 * val + (1 - this.beta2) * Math.pow(grad_bias2[i][j], 2)
      )
    )
    this.v.bias1 = this.v.bias1.map((row, i) =>
      row.map(
        (val, j) =>
          this.beta2 * val + (1 - this.beta2) * Math.pow(grad_bias1[i][j], 2)
      )
    )

    // 更新权重矩阵和偏置矩阵
    this.weights3 = this.adamUpdate(
      this.weights3,
      grad_weights3,
      this.m.weights3,
      this.v.weights3,
      t
    )
    this.weights2 = this.adamUpdate(
      this.weights2,
      grad_weights2,
      this.m.weights2,
      this.v.weights2,
      t
    )
    this.weights1 = this.adamUpdate(
      this.weights1,
      grad_weights1,
      this.m.weights1,
      this.v.weights1,
      t
    )
    this.bias3 = this.adamUpdate(
      this.bias3,
      grad_bias3,
      this.m.bias3,
      this.v.bias3,
      t
    )
    this.bias2 = this.adamUpdate(
      this.bias2,
      grad_bias2,
      this.m.bias2,
      this.v.bias2,
      t
    )
    this.bias1 = this.adamUpdate(
      this.bias1,
      grad_bias1,
      this.m.bias1,
      this.v.bias1,
      t
    )
  }

  /**
   * 训练神经网络
   * @param input 输入矩阵
   * @param target 目标矩阵
   * @param epochs 迭代次数
   * @param batchSize 批次大小
   */
  train(
    input: Matrix,
    target: Matrix,
    epochs: number,
    batchSize: number
  ): void {
    let t = 1
    for (let i = 0; i < epochs; i++) {
      for (let j = 0; j < input.length; j += batchSize) {
        const inputBatch = input.slice(j, j + batchSize)
        const targetBatch = target.slice(j, j + batchSize)
        const [hidden1, hidden2, output] = this.feedForward(inputBatch)
        this.backpropagate(inputBatch, hidden1, hidden2, output, targetBatch, t)
        t++
      }
    }
  }

  /**
   * 预测输出
   * @param input 输入矩阵
   * @returns 输出矩阵
   */
  predict(input: Matrix): Matrix {
    const [_, __, output] = this.feedForward(input)
    return output
  }
}

export default NeuralNetwork
