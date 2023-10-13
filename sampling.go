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
		logits := probs.Clone()
		logits.Log()

		for token, bias := range logitBias {
			logits[token] += bias
		}

		logits.Exp()
		var expLogitsSum = logits.Sum()

		for i := range probs {
			probs[i] = logits[i] / expLogitsSum
		}
	}

	if temperature == 0 {
		return probs.Argmax(), nil
	}

	// 把概率之和 <topP 的那些index过滤出来
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

	if temperature != 1 && temperature > 0 {
		probs.Pow(1.0 / temperature)
	}

	// 这是重新把probs归一化
	var probsSum = probs.Sum()
	probs.Scale(1.0 / probsSum)

	return randomChoice(len(probs), probs), nil
}

func randomChoice(length int, probabilities []float32) int {
	cumulativeProbabilities := make([]float32, length)
	cumulativeProbabilities[0] = probabilities[0]
	for i := 1; i < length; i++ {
		cumulativeProbabilities[i] = cumulativeProbabilities[i-1] + probabilities[i]
	}

	randomValue := rand.Float32()
	for i, cp := range cumulativeProbabilities {
		if randomValue <= cp {
			return i
		}
	}

	return length - 1
}
