import NeuralNetwork from './NeuralNetwork'
import { Matrix } from './utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

// 读取CSV文件数据
const csvData = readFileSync(
  resolve(__dirname, '../dataset/Xor_Dataset.csv'),
  'utf-8'
)
const lines = csvData.split('\n')
const input: Matrix = []
const target: Matrix = []

// 解析CSV数据
for (let i = 1; i < 100; i++) {
  const columns = lines[i].split(',')
  const x1 = parseInt(columns[0])
  const x2 = parseInt(columns[1])
  const xorResult = parseInt(columns[2])
  input.push([x1, x2])
  target.push([xorResult])
}

// 初始化神经网络
const nn = new NeuralNetwork(2, 4, 4, 1, 0.001)

// 计算准确率
function calculateAccuracy(predictions: Matrix, targets: Matrix): number {
  let correct = 0
  predictions.forEach((pred, i) => {
    if (Math.round(pred[0]) === targets[i][0]) {
      correct++
    }
  })
  return (correct / predictions.length) * 100
}

// 训练神经网络
const batchSize = 4
for (let epoch = 0; epoch < 100000; epoch++) {
  nn.train(input, target, 1, batchSize)
  if (epoch % 1000 === 0) {
    const predictions: Matrix = input.map((inp) => nn.predict([inp]).flat())
    const loss =
      target.reduce(
        (sum, t, i) => sum + Math.pow(t[0] - predictions[i][0], 2),
        0
      ) / target.length
    const accuracy = calculateAccuracy(predictions, target)
    console.log(
      `Epoch ${epoch}: Loss = ${loss.toFixed(4)}, Accuracy = ${accuracy.toFixed(
        2
      )}%`
    )
  }
}

const testInput: Matrix = []
const testTarget: Matrix = []

// 解析CSV数据
for (let i = 100; i < 200; i++) {
  const columns = lines[i].split(',')
  const x1 = parseInt(columns[0])
  const x2 = parseInt(columns[1])
  const xorResult = parseInt(columns[2])
  testInput.push([x1, x2])
  testTarget.push([xorResult])
}

// 输出预测结果
testInput.forEach((inp, index) => {
  const output = nn.predict([inp])
  console.log(
    `Input: ${inp}, Predicted: ${output[0][0]}, Target: ${testTarget[index][0]}`
  )
})
