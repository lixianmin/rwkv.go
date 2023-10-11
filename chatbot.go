package rwkv

import (
	"fmt"
	"golang.org/x/exp/slices"
	"math"
	"strings"
)

/********************************************************************
created:    2023-10-09
author:     lixianmin

Copyright (C) - All Rights Reserved
*********************************************************************/

const (
	GEN_TEMPERATURE = 1.2 //It could be a good idea to increase temp when top_p is low
	GEN_TOP_P       = 0.5 //Reduce top_p (to 0.5, 0.2, 0.1 etc.) for better Q&A accuracy (and less diversity)

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
}

func NewChatbot(model *ChatModel, userName string, botName string, prompt string) *Chatbot {
	var avoidRepeatTokens = model.Encode(AVOID_REPEAT)
	var chatbot = &Chatbot{
		model:             model,
		userName:          userName,
		botName:           botName,
		avoidRepeatTokens: avoidRepeatTokens,
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

	for i := 0; i < 999; i++ {
		var newlineAdj float32 = 0
		if i <= 0 {
			newlineAdj = -999999999
		} else if i <= chatLenShort {
			newlineAdj = float32(i-chatLenShort) * 0.1
		} else if i <= chatLenLong {
			newlineAdj = 0
		} else {
			newlineAdj = float32(math.Min(3.0, float64(i-chatLenLong)*0.25)) // MUST END THE GENERATION
		}

		for k, v := range existing {
			logits[k] -= GEN_alpha_presence + v*GEN_alpha_frequency
		}

		var token, _ = SampleLogits(logits, GEN_TEMPERATURE, GEN_TOP_P, nil)
		for xxx := range existing {
			existing[xxx] *= GEN_penalty_decay
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

		var sendMessage = my.model.Decode(tokens)
		if strings.Contains(sendMessage, "\n\n") {
			break
		}
	}

	var output = strings.Join(pieces, "")
	return output
}
