package rwkv

import (
	"errors"
	"log"
	"os"
	"strings"
)

/********************************************************************
created:    2023-10-09
author:     lixianmin

Copyright (C) - All Rights Reserved
*********************************************************************/

type ChatModel struct {
	dylibPath string
	cRwkv     CRwkv
	options   *RwkvOptions
	tokenizer Tokenizer
	ctx       *RwkvCtx
}

func NewChatModel(modelPath string, options RwkvOptions) (*ChatModel, error) {
	file, err := dumpRwkvLibrary(options.GpuEnable)
	if err != nil {
		return nil, err
	}

	dylibPath := file.Name()

	cRwkv, err := NewCRwkv(dylibPath)
	if err != nil {
		return nil, err
	}

	var tk Tokenizer

	if options.TokenizerType == Normal {
		tk, err = NewNormalTokenizer()
	} else if options.TokenizerType == World {
		tk, err = NewWorldTokenizer()
	}

	if err != nil {
		return nil, err
	}

	if options.GpuEnable {
		log.Printf("You are about to offload your model to the GPU. " +
			"Please confirm the size of your GPU memory to prevent memory overflow." +
			"If the model is larger than GPU memory, please specify the layers to offload.")
	}

	var model = &ChatModel{
		dylibPath: dylibPath,
		cRwkv:     cRwkv,
		options:   &options,
		tokenizer: tk,
	}

	var err2 = model.loadFromFile(modelPath)
	if err2 != nil {
		return nil, err2
	}

	return model, nil
}

func (my *ChatModel) loadFromFile(path string) error {
	_, err := os.Stat(path)
	if err != nil {
		return errors.New("the system cannot find the model file specified")
	}

	var ctx = my.cRwkv.RwkvInitFromFile(path, my.options.CpuThreads)
	var err2 = hasCtx(ctx)
	if err2 != nil {
		return err2
	}

	my.ctx = ctx
	// offload all layers to GPU
	gpuNLayers := uint32(my.cRwkv.RwkvGetNLayer(ctx) + 1)
	// if user specify the layers to offload, use the user specified value
	if my.options.GpuOffLoadLayers > 0 {
		gpuNLayers = my.options.GpuOffLoadLayers
	}

	if my.options.GpuEnable {
		err = my.cRwkv.RwkvGpuOffloadLayers(ctx, gpuNLayers)
		if err != nil {
			return err
		}
	}

	// by default disable error printing and handle errors by go error
	my.cRwkv.RwkvSetPrintErrors(ctx, my.options.PrintError)
	return nil
}

func (my *ChatModel) Encode(input string) ([]int, error) {
	return my.tokenizer.Encode(input)
}

func (my *ChatModel) Decode(input []int) string {
	return my.tokenizer.Decode(input)
}

func (my *ChatModel) Eval(tokens []int) (string, error) {
	var state = make([]float32, my.cRwkv.RwkvGetStateLength(my.ctx))
	my.cRwkv.RwkvInitState(my.ctx, state)
	var logits = make([]float32, my.cRwkv.RwkvGetLogitsLength(my.ctx))

	for _, token := range tokens {
		var err = my.cRwkv.RwkvEval(my.ctx, uint32(token), state, state, logits)
		if err != nil {
			return "", err
		}
	}

	return my.generateResponse(state, logits)
}

func (my *ChatModel) generateResponse(state, logits []float32) (string, error) {
	var responseText = ""
	var options = my.options

	for i := 0; i < options.MaxTokens; i++ {
		token, err := SampleLogits(logits, options.Temperature, options.TopP, map[int]float32{})
		if err != nil {
			return "", err
		}

		err = my.cRwkv.RwkvEval(my.ctx, uint32(token), state, state, logits)
		if err != nil {
			return "", err
		}

		chars := my.tokenizer.Decode([]int{token})
		responseText += chars

		if strings.Contains(responseText, options.StopString) {
			responseText = strings.Split(responseText, options.StopString)[0]
			break
		}
	}

	return responseText, nil
}
