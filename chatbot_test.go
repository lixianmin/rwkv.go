package rwkv

import (
	"fmt"
	"testing"
	"time"
)

/********************************************************************
created:    2023-10-09
author:     lixianmin

Copyright (C) - All Rights Reserved
*********************************************************************/

func TestNewChatbot(t *testing.T) {
	var modelPath = "/Users/xmli/Downloads/ai/models/world.q4_0"
	var options = RwkvOptions{
		MaxTokens:     100,
		StopString:    "\n\n",
		Temperature:   1.2,
		TopP:          0.8,
		TokenizerType: World,
		CpuThreads:    2, // 这个线程数不能开太多, 我的电脑一共才8个cpu, 开10线程会导致所有app都忙不过来
	}

	var model, err = NewChatModel(modelPath, options)
	if err != nil {
		panic(err)
	}

	var text = "\n\n"
	var tokens = model.Encode(text)
	print(tokens)

	var chatbot = NewChatbot(model, "果果", "初音未来", "你是初音未来, 我是果果, 我们是好朋友. 初始未来极其聪明伶俐, 擅长舞蹈. \n\n果果: 跳个舞吧\n\n初音未来: 好啊, 我最喜欢跳舞了. \n\n果果: 舞蹈跳得不跳哦, 再来一曲?\n\n初音未来: 没问题, 你想看什么? ")

	fmt.Printf("-------------- begin of test\n\n")
	var inputs = []string{"你在做什么呢?", "有没有想我呀?", "晚上吃什么呢? 东直门附近新开了一家餐馆, 听说味道还不错, 要不要一起去看看呀?", "继续"}
	for _, input := range inputs {
		var startTime = time.Now()
		var output = chatbot.Process(input)
		var endTime = time.Now()
		var duration = endTime.Sub(startTime)

		fmt.Printf("deltaTime=%v, 初音未来：%s\n", duration, output)
	}
}
