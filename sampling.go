// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

import (
	"errors"
	"github.com/chewxy/math32"
	"gorgonia.org/vecf32"
	"math/rand"
	"slices"
	"sort"
	"time"
)

func softmax(out []float32) []float32 {
	maxVal := slices.Max(out)
	expSum := float32(0.0)
	for i := range out {
		out[i] = math32.Exp(out[i] - maxVal)
		expSum += out[i]
	}

	vecf32.Scale(out, 1.0/expSum)
	return out
}

func SampleLogits(tensor []float32, temperature float32, topP float32, logitBias map[int]float32) (int, error) {
	if temperature < 0 {
		return 0, errors.New("temperature must be non-negative")
	}
	if topP < 0 || topP > 1 {
		return 0, errors.New("top_p must be in the range [0, 1]")
	}

	if topP == 0 {
		topP = 1
	}

	probs := softmax(tensor)
	return sampleProbs(probs, temperature, topP, logitBias)
}

func sampleProbs(probs []float32, temperature float32, topP float32, logitBias map[int]float32) (int, error) {
	if logitBias != nil {
		logits := slices.Clone(probs)
		for i := range logits {
			logits[i] = math32.Log(logits[i])
		}

		for token, bias := range logitBias {
			logits[token] += bias
		}

		expLogitsSum := float32(0.0)
		for i := range logits {
			logits[i] = math32.Exp(logits[i])
			expLogitsSum += logits[i]
		}

		for i := range probs {
			probs[i] = logits[i] / expLogitsSum
		}
	}

	if temperature == 0 {
		return vecf32.Argmax(probs), nil
	}

	// 把概率之和 <topP 的那些index过滤出来
	if topP < 1 {
		var sortedProbs = slices.Clone(probs)
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
		var invTemperature = 1.0 / temperature
		for i := range probs {
			probs[i] = math32.Pow(probs[i], invTemperature)
		}
	}

	var probsSum = vecf32.Sum(probs)
	vecf32.Scale(probs, 1.0/probsSum)

	return randomChoice(len(probs), probs), nil
}

func randomChoice(length int, probabilities []float32) int {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	cumulativeProbabilities := make([]float32, length)
	cumulativeProbabilities[0] = probabilities[0]
	for i := 1; i < length; i++ {
		cumulativeProbabilities[i] = cumulativeProbabilities[i-1] + probabilities[i]
	}

	randomValue := r.Float32()
	for i, cp := range cumulativeProbabilities {
		if randomValue <= cp {
			return i
		}
	}

	return length - 1
}
