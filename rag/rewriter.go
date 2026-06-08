package rag

import (
	"context"
	"fmt"
	"rag-course/llm"
	"strings"
)

const rewriteSystemPrompt = `您将用户的最新消息重写为独立的搜索查询。 

给定对话，输出单个搜索查询： 
- 捕获最新用户消息的主题意图。
- 使用先前的回合（“it”、“they”、“that one”）解析代词和指称。
- 保持简洁 - 关键词和短语，而不是完整的句子。

如果最新的用户消息已经独立存在，没有引用之前的轮次，则逐字输出。

仅输出查询。 没有序言，没有引文，没有解释。`

type Rewriter struct {
	client *llm.Client
}

func NewRewriter(client *llm.Client) *Rewriter {
	return &Rewriter{
		client: client,
	}
}

// Rewrite 分析当前的对话历史，利用 LLM 将用户的最新发言重写为一个独立、明确的搜索查询词。
// 这一步对于多轮对话 RAG 非常关键，因为用户的后续提问可能包含代词（如“它”、“这个”）或省略主语，
// 只有结合上下文进行指代消解后，才能在向量数据库中准确检索。
func (r *Rewriter) Rewrite(ctx context.Context, history []llm.Message) (string, error) {
	// 获取用户最后一次提问
	last := lastUserMessage(history)
	if last == "" {
		return "", nil
	}

	// 如果历史记录中没有 AI 的回复，说明是第一轮对话，此时提问本身就是完整的，不需要重写。
	if !hasAssistantTurn(history) {
		return last, nil
	}

	// 构造专门用于重写的 LLM 请求
	msgs := []llm.Message{
		{Role: "system", Content: rewriteSystemPrompt},
		{Role: "user", Content: formatConversation(history)},
	}

	// 调用 LLM 执行重写（这里禁用了流式输出）
	reply, err := r.client.ChatStream(ctx, msgs, nil)
	if err != nil {
		return "", fmt.Errorf("rewrite call: %w", err)
	}

	// 清理模型输出，去除两端的空白字符和引号
	out := strings.TrimSpace(reply.Content)
	out = strings.Trim(out, `"'`)
	if out == "" {
		// 如果 LLM 未能生成有效的重写查询，回退到使用原始的用户提问
		return last, nil
	}

	return out, nil
}

func hasAssistantTurn(history []llm.Message) bool {
	for _, m := range history {
		if m.Role == "assistant" {
			return true
		}
	}
	return false
}

func formatConversation(history []llm.Message) string {
	var sb strings.Builder
	sb.WriteString("目前的对话:\n\n")
	for _, m := range history {
		switch m.Role {
		case "user":
			sb.WriteString("User: ")
		case "assistant":
			sb.WriteString("Assistant:")
		default:
			continue
		}
		sb.WriteString(m.Content)
		sb.WriteString("\n\n")
	}
	sb.WriteString("将用户的最新消息重写为独立的搜索查询。")
	return sb.String()
}
