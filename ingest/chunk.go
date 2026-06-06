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

	// 2. 如果文本总长度本身就小于或等于目标分块大小 size，直接作为一个整体分块返回
	if len(text) <= size {
		return []string{text}
	}

	// 3. 校验并调整重叠区间大小 overlap
	if overlap < 0 {
		overlap = 0
	}
	// 重叠区间不能过大，如果超过或等于分块大小，强制限制为分块大小的一半
	if overlap >= size {
		overlap = size / 2
	}

	// 4. 设定自然分割符的查找阈值（当前分块窗口大小的 70%）
	// 只有当分割符出现在窗口后半段（>= 70%）时，才在此处切分，避免分块过小
	threshold := size * 7 / 10

	var chunks []string
	n := len(text)
	start := 0

	// 5. 循环滑动窗口对文本进行切块
	for start < n {
		end := start + size
		// 5.1 如果当前窗口终点超过了文本的总长度，说明已到达文本尾部
		if end > n {
			if part := strings.TrimSpace(text[start:]); part != "" {
				chunks = append(chunks, part)
			}
			break
		}

		// 5.2 获取当前窗口范围内的文本段
		window := text[start:end]

		// 5.3 智能寻找切分点：在窗口内尽可能寻找自然的停顿/边界符，避免粗暴截断单词或句子
		// 优先级：段落(双换行) > 句子结束(句号加空格) > 单词间隔(空格)
		switch {
		// 寻找最后出现的双换行符 "\n\n"（通常是段落边界）
		case strings.LastIndex(window, "\n\n") >= threshold:
			end = start + strings.LastIndex(window, "\n\n") + 2
		// 寻找最后出现的句号加空格 ". "（通常是句子边界）
		case strings.LastIndex(window, ". ") >= threshold:
			end = start + strings.LastIndex(window, ". ") + 2
		// 寻找最后出现的中文句号 "。"（通常是句子边界，UTF-8 占 3 字节）
		case strings.LastIndex(window, "。") >= threshold:
			end = start + strings.LastIndex(window, "。") + 3
		// 寻找最后出现的空格 " "（通常是单词/英文单词边界）
		case strings.LastIndex(window, " ") >= threshold:
			end = start + strings.LastIndex(window, " ") + 1
		}

		// 5.4 截取确定的分块，去除多余首尾空格后存入结果中
		part := strings.TrimSpace(text[start:end])
		if part != "" {
			chunks = append(chunks, part)
		}

		// 5.5 更新下一次滑动的起始位置（当前终点减去重叠区大小）
		start = end - overlap
	}
	return chunks
}
