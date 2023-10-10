// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

import (
	"bufio"
	"embed"
	"errors"
	"strconv"
	"strings"
	"unicode/utf8"
)

//go:embed rwkv_vocab_v20230424.txt
var worldTokenizerFS embed.FS

// Trie represents the trie data structure
type Trie struct {
	ch     byte
	to     []*Trie
	values map[int]byte
	front  *Trie
}

func NewTrie(front *Trie, ch byte) *Trie {
	var trie = &Trie{
		ch:     ch,
		to:     make([]*Trie, 256),
		values: make(map[int]byte),
		front:  front,
	}

	return trie
}

func (my *Trie) Add(key string, idx int, val int) *Trie {
	if idx == len(key) {
		my.values[val] = 0
		return my
	}

	var ch = key[idx]
	if my.to[ch] == nil {
		my.to[ch] = NewTrie(my, ch)
	}

	return my.to[ch].Add(key, idx+1, val)
}

func (my *Trie) FindLongest(key string, idx int) (int, *Trie, map[int]byte) {
	var u = my
	var ch = key[idx]

	var retIndex int
	var retTrie *Trie
	var retValues map[int]byte

	for u.to[ch] != nil {
		u = u.to[ch]
		idx += 1

		if len(u.values) != 0 {
			retIndex = idx
			retTrie = u
			retValues = u.values
		}

		if idx == len(key) {
			break
		}

		ch = key[idx]
	}

	return retIndex, retTrie, retValues
}

// WorldTokenizer represents a tokenizer for encoding and decoding bytes to tokens
type WorldTokenizer struct {
	IndexToToken map[int]string
	TokenToIndex map[string]int
	Trie         *Trie
}

// NewWorldTokenizer initializes a new world tokenizer
func NewWorldTokenizer() (*WorldTokenizer, error) {
	f, err := worldTokenizerFS.Open("rwkv_vocab_v20230424.txt")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	wt := &WorldTokenizer{
		IndexToToken: make(map[int]string),
		TokenToIndex: make(map[string]int),
		Trie:         NewTrie(nil, 0),
	}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		leftIndex := strings.Index(line, " ")
		rightIndex := strings.LastIndex(line, " ")

		index, err := strconv.Atoi(line[:leftIndex])
		if err != nil {
			return nil, err
		}

		var size, err2 = strconv.Atoi(line[rightIndex+1:])
		if err2 != nil {
			return nil, err2
		}

		var input = line[leftIndex+1 : rightIndex]
		token, err := parseInput(input, size)
		if err != nil {
			return nil, err
		}

		wt.IndexToToken[index] = token
		wt.TokenToIndex[token] = index
		wt.Trie.Add(token, 0, index)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return wt, nil
}

// EncodeBytes encodes bytes to tokens
func (wt *WorldTokenizer) EncodeBytes(src string) ([]int, error) {
	var tokens []int
	idx := 0
	for idx < len(src) {
		var values map[int]byte
		idx, _, values = wt.Trie.FindLongest(src, idx)
		for token := range values {
			tokens = append(tokens, token)
			break
		}
	}

	return tokens, nil
}

// DecodeBytes decodes tokens to bytes
func (wt *WorldTokenizer) DecodeBytes(tokens []int) []byte {
	var result []byte
	for _, token := range tokens {
		result = append(result, []byte(wt.IndexToToken[token])...)
	}
	return result
}

// Encode encodes a string to tokens
func (wt *WorldTokenizer) Encode(src string) ([]int, error) {
	return wt.EncodeBytes(src)
}

// Decode decodes tokens to a string
func (wt *WorldTokenizer) Decode(tokens []int) string {
	return string(wt.DecodeBytes(tokens))
}

func parseInput(input string, size int) (string, error) {
	var text = input
	var isBinary = input[0] == 'b'
	if isBinary {
		text = text[1:]
	}

	if strings.HasPrefix(text, "'") && strings.HasSuffix(text, "'") {
		text = strings.ReplaceAll(text, "\"", "\\\"")
		text = "\"" + text[1:len(text)-1] + "\""
		text = strings.ReplaceAll(text, "\\'", "'")
	}

	var raw, err1 = strconv.Unquote(text)
	if err1 != nil {
		return "", err1
	}

	var word = raw
	var rawBytes = []byte(raw)
	if len(rawBytes) != size {
		var encoded = encode2UTF8(rawBytes)
		word = string(encoded)

		if len(encoded) != size {
			return "", errors.New("invalid input")
		}
	}

	return word, nil
}

func encode2UTF8(input []byte) []byte {
	var buff = make([]byte, len(input)*utf8.UTFMax)
	var length = 0
	for _, b := range input {
		length += utf8.EncodeRune(buff[length:], rune(b))
	}

	return buff[:length]
}
