// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

import (
	"errors"
	"github.com/lixianmin/v32"
	"math/rand"
	"sort"
)

func SampleLogits(logits v32.V32, temperature float32, topP float32, logitBias map[int]float32) (int, error) {
	if temperature < 0 {
		return 0, errors.New("temperature must be non-negative")
	}
	if topP < 0 || topP > 1 {
		return 0, errors.New("top_p must be in the range [0, 1]")
	}

	if topP == 0 {
		topP = 1
	}

	logits.SoftMax()
	return sampleProbs(logits, temperature, topP, logitBias), nil
}

func sampleProbs(probs v32.V32, temperature float32, topP float32, logitBias map[int]float32) int {
	if logitBias != nil {
		// 这段代码因为从来未用到, 所以先保持. 但看起来这个Clone()是没有意义的, 直接使用probs[]就好
		var cloned = probs.Clone()
		cloned.Log()

		for token, bias := range logitBias {
			cloned[token] += bias
		}

		cloned.Exp()
		var invSum = 1.0 / cloned.Sum()

		for i := range probs {
			probs[i] = cloned[i] * invSum
		}
	}

	if temperature == 0 {
		return probs.Argmax()
	}

	// 把概率之和 <topP 的那些index过滤出来
	filterTopP(probs, topP)

	// temperature过大, 会导致probs里的数值打平为1, 这样所有的备选的概率就都一样了.
	// temperature过小会导致重复, temperature过大会导致胡说八道
	// https://zhuanlan.zhihu.com/p/613428710
	if temperature != 1 && temperature > 0 {
		probs.Pow(1.0 / temperature)
	}

	// v1. 类似于重新把probs归一化, 使其所有成员的和等于1.0. 这不能省, 因为pow()修改probs[]的值
	probs.Scale(1.0 / probs.Sum())
	return randomChoice(probs)
}

func filterTopP(probs v32.V32, topP float32) float32 {
	var realTopP = topP
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
					realTopP = topP
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

	return realTopP
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
