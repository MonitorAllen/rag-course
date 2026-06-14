package rag

import (
	"fmt"
	"rag-course/vector"
	"strings"
)

const contextPreamble = `请使用以下文献集的摘录来回答这个问题。引用来源时，请按文件名引用。如果摘录内容没有回答问题，请在基于常识回答前说明。`

const unknownSource = "(未知来源)"

// formatContext 将从向量数据库中检索到的多条文档切片 (hits) 组装成纯文本的上下文段落。
// 这段文本将被注入到用户提问的前面，引导大语言模型基于这些特定文献的内容进行回答。
func formatContext(hits []vector.Result) string {
	if len(hits) == 0 {
		return ""
	}

	var sb strings.Builder
	// 添加系统前置提示，限制模型行为：要求它基于提供的摘要回答问题，并说明如何处理信息不足的情况。
	sb.WriteString(contextPreamble)
	// （这里的“常识回答”占位符可以视为分隔符，但在前面的 preamble 中提示了如何 fallback）
	sb.WriteString("\n\n--- 常识回答 --- \n\n")

	// 遍历所有匹配的文档，附加索引、来源、相关度得分以及具体内容
	for i, h := range hits {
		source := h.Metadata["source"]
		if source == "" {
			source = unknownSource
		}

		if h.Metadata["type"] == "image" && h.Metadata["image_path"] != "" {
			imagePath := h.Metadata["image_path"]
			fmt.Fprintf(&sb, "[%d] 来源: %s [图片: %s] (相关性 %.2f)\n%s\n\n", i+1, source, imagePath, h.Score, h.Content)
			continue
		}

		fmt.Fprintf(&sb, "[%d] 来源: %s (相关性 %.2f)\n%s\n\n", i+1, source, h.Score, h.Content)
	}

	return strings.TrimSpace(sb.String())
}
