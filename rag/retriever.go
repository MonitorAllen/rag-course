package rag

import (
	"context"
	"fmt"
	"rag-course/llm"
	"rag-course/vector"
)

const defaultTopK = 5

type Options struct {
	TopK     int
	Rewriter *Rewriter
}

type Retriever struct {
	embedder llm.Embedder
	store    vector.Store
	rewriter *Rewriter
	topK     int
}

func New(embedder llm.Embedder, store vector.Store, opts Options) *Retriever {
	topK := opts.TopK
	if topK <= 0 {
		topK = defaultTopK
	}

	return &Retriever{
		embedder: embedder,
		store:    store,
		rewriter: opts.Rewriter,
		topK:     topK,
	}
}

// Retrieve 接收对话历史，根据用户的提问或者重写后的独立查询去向量数据库中检索相关文档。
// 返回值为拼接好的上下文文本，如果检索不到则返回空字符串。
func (r *Retriever) Retrieve(ctx context.Context, history []llm.Message) (string, error) {
	// 1. 基于对话历史构建用于检索的独立查询文本
	query := r.buildQuery(ctx, history)
	if query == "" {
		return "", nil
	}

	// 2. 将查询文本转化为向量 (Embedding)，以便于在向量数据库中进行语义相似度检索
	vecs, err := r.embedder.Embed(ctx, []string{query})
	if err != nil {
		return "", fmt.Errorf("embed query: %w", err)
	}
	if len(vecs) == 0 {
		return "", nil
	}

	// 3. 在向量数据库中查询最相似的 topK 个文档切片 (hits)
	hits, err := r.store.Query(ctx, vecs[0], r.topK)
	if err != nil {
		return "", fmt.Errorf("vector query: %w", err)
	}
	if len(hits) == 0 {
		return "", nil
	}

	// 4. 将检索到的结果格式化为包含来源和内容的纯文本
	return formatContext(hits), nil
}

// buildQuery 基于当前的对话历史来构造搜索查询词。
// 如果启用了重写器 (rewriter)，它会将带有指代代词（例如“它”）的消息重写为独立的搜索词。
// 否则，它仅仅使用用户最新的一次发言作为检索词。
func (r *Retriever) buildQuery(ctx context.Context, history []llm.Message) string {
	if r.rewriter != nil {
		// 尝试使用 LLM 进行提问重写
		if q, err := r.rewriter.Rewrite(ctx, history); err == nil && q != "" {
			return q
		}
	}
	return lastUserMessage(history)
}

func lastUserMessage(history []llm.Message) string {
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == "user" {
			return history[i].Content
		}
	}
	return ""
}
