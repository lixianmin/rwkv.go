// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

import (
	"errors"
	"github.com/lixianmin/v32"
	"math/rand"
	"sort"
)

func SampleLogits(tensor v32.V32, temperature float32, topP float32, logitBias map[int]float32) (int, error) {
	if temperature < 0 {
		return 0, errors.New("temperature must be non-negative")
	}
	if topP < 0 || topP > 1 {
		return 0, errors.New("top_p must be in the range [0, 1]")
	}

	if topP == 0 {
		topP = 1
	}

	tensor.SoftMax()
	return sampleProbs(tensor, temperature, topP, logitBias)
}

func sampleProbs(probs v32.V32, temperature float32, topP float32, logitBias map[int]float32) (int, error) {
	if logitBias != nil {
		var logits = probs.Clone()
		logits.Log()

		for token, bias := range logitBias {
			logits[token] += bias
		}

		logits.Exp()
		var invSum = 1.0 / logits.Sum()

		for i := range probs {
			probs[i] = logits[i] * invSum
		}
	}

	if temperature == 0 {
		return probs.Argmax(), nil
	}

	// 把概率之和 <topP 的那些index过滤出来
	filterTopP(probs, topP)

	if temperature != 1 && temperature > 0 {
		probs.Pow(1.0 / temperature)
	}

	// v1. 类似于重新把probs归一化, 使其所有成员的和等于1.0
	// v2. 在randomChoice()中把random数值乘以realTopP, 不再需要归一化处理了
	// v3. 归一化这事不能省, 作为logits的probs是要传出去, 用于其它地方的计算的
	var probsSum = probs.Sum()
	probs.Scale(1.0 / probsSum)

	return randomChoice(probs), nil
}

func filterTopP(probs v32.V32, topP float32) {
	if topP < 1 {
		var sortedProbs = probs.Clone()
		sort.Slice(sortedProbs, func(i, j int) bool { return sortedProbs[i] > sortedProbs[j] })

		cutoff := sortedProbs[0]
		if sortedProbs[0] < topP {
			for i := 1; i < len(sortedProbs); i++ {
				var last = sortedProbs[i]
				sortedProbs[i] = sortedProbs[i-1] + last
				if sortedProbs[i] > topP { // 因为 topP<1, 因此这个条件在循环过程中一定是有机会成立的
					cutoff = last
					break
				}
			}
		}

		for i, p := range probs {
			if p < cutoff {
				probs[i] = 0
			}
		}
	}
}

func randomChoice(probs v32.V32) int {
	var sum = float32(0)
	var random = rand.Float32()
	for i, p := range probs {
		sum += p
		if random <= sum {
			return i
		}
	}

	return len(probs) - 1
}
