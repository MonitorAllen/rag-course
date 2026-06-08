package ingest

import (
	"strings"
)

func chunk(text string, size, overlap int) []string {
	// 1. 去除文本首尾的多余空白字符
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	// 使用 []rune 避免按字节切分导致中文字符（多字节）被截断
	runes := []rune(text)

	// 2. 如果文本总长度本身就小于或等于目标分块大小 size，直接作为一个整体分块返回
	if len(runes) <= size {
		return []string{text}
	}

	// 3. 校验并调整重叠区间大小 overlap
	if overlap < 0 {
		overlap = 0
	}
	if overlap >= size {
		overlap = size / 2
	}

	// 4. 设定自然分割符的查找阈值（当前分块窗口大小的 70%）
	threshold := size * 7 / 10

	var chunks []string
	n := len(runes)
	start := 0

	// 5. 循环滑动窗口对文本进行切块
	for start < n {
		end := start + size
		// 5.1 如果当前窗口终点超过了文本的总长度，说明已到达文本尾部
		if end > n {
			if part := strings.TrimSpace(string(runes[start:])); part != "" {
				chunks = append(chunks, part)
			}
			break
		}

		// 5.2 智能寻找切分点：在窗口后半部分寻找自然停顿边界，优先级：段落 > 英文句子 > 中文句子 > 空格
		idxDoubleNewline := -1
		idxDotSpace := -1
		idxChinesePeriod := -1
		idxSpace := -1

		for i := start + threshold; i < end; i++ {
			if i > start && runes[i-1] == '\n' && runes[i] == '\n' {
				idxDoubleNewline = i + 1
			}
			if i > start && runes[i-1] == '.' && runes[i] == ' ' {
				idxDotSpace = i + 1
			}
			if runes[i] == '。' {
				idxChinesePeriod = i + 1
			}
			if runes[i] == ' ' {
				idxSpace = i + 1
			}
		}

		if idxDoubleNewline != -1 {
			end = idxDoubleNewline
		} else if idxDotSpace != -1 {
			end = idxDotSpace
		} else if idxChinesePeriod != -1 {
			end = idxChinesePeriod
		} else if idxSpace != -1 {
			end = idxSpace
		}

		// 5.4 截取确定的分块，去除多余首尾空格后存入结果中
		part := strings.TrimSpace(string(runes[start:end]))
		if part != "" {
			chunks = append(chunks, part)
		}

		// 5.5 更新下一次滑动的起始位置（当前终点减去重叠区大小）
		start = end - overlap
	}
	return chunks
}
