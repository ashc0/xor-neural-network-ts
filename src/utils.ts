export type Matrix = number[][]

// 生成一个随机矩阵
export function randomMatrix(rows: number, cols: number): Matrix {
  const matrix: Matrix = []
  for (let i = 0; i < rows; i++) {
    matrix.push(
      Array(cols)
        .fill(0)
        .map(() => Math.random() - 0.5)
    )
  }
  return matrix
}

// 生成一个Xavier初始化的矩阵
export function xavierMatrix(rows: number, cols: number): Matrix {
  const scale = Math.sqrt(2.0 / (rows + cols))
  const matrix: Matrix = []
  for (let i = 0; i < rows; i++) {
    matrix.push(
      Array(cols)
        .fill(0)
        .map(() => (Math.random() * 2 - 1) * scale)
    )
  }
  return matrix
}

// 生成一个He初始化的矩阵
export function heMatrix(rows: number, cols: number): Matrix {
  const scale = Math.sqrt(2.0 / (rows + cols))
  const matrix: Matrix = []
  for (let i = 0; i < rows; i++) {
    matrix.push(
      Array(cols)
        .fill(0)
        .map(() => (Math.random() * 2 - 1) * scale)
    )
  }
  return matrix
}

// 生成一个全零矩阵
export function zerosMatrix(rows: number, cols: number): Matrix {
  return Array(rows)
    .fill(null)
    .map(() => Array(cols).fill(0))
}

// sigmoid函数
export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

// sigmoid函数的导数
export function sigmoidDerivative(x: number): number {
  const s = sigmoid(x)
  return s * (1 - s)
}

// ReLU函数
export function relu(x: number): number {
  return Math.max(0, x)
}

// ReLU函数的导数
export function reluDerivative(x: number): number {
  return x > 0 ? 1 : 0
}

// tanh函数
export function tanh(x: number): number {
  return Math.tanh(x)
}

/**
 * tanh函数的导数
 */
export function tanhDerivative(x: number): number {
  const t = tanh(x)
  return 1 - t * t
}

/**
 * 矩阵乘法
 */
export function dot(a: Matrix, b: Matrix): Matrix {
  const result: Matrix = zerosMatrix(a.length, b[0].length)
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b[0].length; j++) {
      for (let k = 0; k < a[0].length; k++) {
        result[i][j] += a[i][k] * b[k][j]
      }
    }
  }
  return result
}

/**
 * 矩阵转置
 */
export function transpose(matrix: Matrix): Matrix {
  return matrix[0].map((_, colIndex) => matrix.map((row) => row[colIndex]))
}
