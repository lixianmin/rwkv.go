package rwkv

import (
	"fmt"
	"slices"
	"strings"
)

/********************************************************************
created:    2023-10-09
author:     lixianmin

Copyright (C) - All Rights Reserved
*********************************************************************/

const (
	GEN_alpha_presence  = 0.4 // Presence Penalty
	GEN_alpha_frequency = 0.4 // Frequency Penalty
	GEN_penalty_decay   = 0.996

	END_OF_TEXT = 0
	END_OF_LINE = 11

	AVOID_REPEAT = "，：？！"
)

type Chatbot struct {
	model             *ChatModel
	userName          string
	botName           string
	avoidRepeatTokens []int
	promptState       []float32
	stopTexts         []string
}

func NewChatbot(model *ChatModel, userName string, botName string, prompt string) *Chatbot {
	var avoidRepeatTokens = model.Encode(AVOID_REPEAT)
	var chatbot = &Chatbot{
		model:             model,
		userName:          userName,
		botName:           botName,
		avoidRepeatTokens: avoidRepeatTokens,
		stopTexts:         []string{"\n\n", userName + ": ", botName + ": "}, // 目前固定使用英文的:来进行分割讲话
	}

	_ = chatbot.initPrompt(prompt)
	return chatbot
}

func (my *Chatbot) initPrompt(prompt string) error {
	var tokens = my.model.Encode(prompt)
	var state, _ = my.runRnn(tokens, nil, 0)
	my.promptState = state
	return nil
}

func (my *Chatbot) runRnn(tokens []int, state []float32, newlineAdj float32) ([]float32, []float32) {
	state, logits := my.model.EvalSequence(tokens, state)
	logits[END_OF_LINE] += newlineAdj

	var lastIndex = len(tokens) - 1
	var last = tokens[lastIndex]

	if slices.Contains(my.avoidRepeatTokens, last) {
		logits[lastIndex] = -999999999
	}

	return state, logits
}

func (my *Chatbot) Process(message string) string {
	message = strings.ReplaceAll(message, "\r\n", "\n")
	message = strings.ReplaceAll(message, "\\n", "\n")
	message = strings.TrimSpace(message)

	var current = fmt.Sprintf("%s: %s\n\n%s: ", my.userName, message, my.botName)
	var output = my.generate(current)
	return output
}

func (my *Chatbot) generate(text string) string {
	var tokens = my.model.Encode(text)
	var state = slices.Clone(my.promptState)
	state, logits := my.runRnn(tokens, state, -999999999)

	tokens = tokens[:0]
	var outLast = 0
	var existing = make(map[int]float32)

	var chatLenShort = 40
	var chatLenLong = 150
	var pieces = make([]string, 0, 16)
	var stopText = ""

	var options = my.model.options

	for i := 0; i < options.MaxTokens; i++ {
		var newlineAdj float32 = 0
		if i <= 0 {
			newlineAdj = -999999999
		} else if i <= chatLenShort {
			newlineAdj = float32(i-chatLenShort) * 0.1
		} else if i <= chatLenLong {
			newlineAdj = 0
		} else {
			newlineAdj = min(3.0, float32(i-chatLenLong)*0.25) // MUST END THE GENERATION
		}

		for k, v := range existing {
			logits[k] -= GEN_alpha_presence + v*GEN_alpha_frequency
		}

		var token, _ = SampleLogits(logits, options.Temperature, options.TopP, nil)
		for t := range existing {
			existing[t] *= GEN_penalty_decay
		}

		existing[token] += 1
		tokens = append(tokens, token)

		state, logits = my.runRnn([]int{token}, state, newlineAdj)
		logits[END_OF_TEXT] = -999999999 // disable <|endoftext|>

		var piece = my.model.Decode(tokens[outLast:])
		if !strings.Contains(piece, "\ufffd") {
			pieces = append(pieces, piece)
			outLast = i + 1
		}

		stopText = my.meetStopText(pieces)
		if stopText != "" {
			break
		}
	}

	// 移除stopText尾巴
	var output = strings.Join(pieces, "")
	if len(stopText) != 0 {
		output = output[:len(output)-len(stopText)]
	}

	return output
}

// 发现stopText尾巴, 就代表要结束了
func (my *Chatbot) meetStopText(pieces []string) string {
	for _, stopText := range my.stopTexts {
		if isEndsWith(pieces, stopText) {
			return stopText
		}
	}

	return ""
}

// 判断pieces是否以stopText结尾
func isEndsWith(pieces []string, stopText string) bool {
	var k = len(stopText) - 1
	for i := len(pieces) - 1; i >= 0; i-- {
		var piece = pieces[i]
		for j := len(piece) - 1; j >= 0; j-- {
			if piece[j] != stopText[k] {
				return false
			} else if k == 0 {
				return true
			}

			k--
		}
	}

	return false
}
