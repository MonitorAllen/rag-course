package llm

import (
	"context"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
)

type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	params := openai.EmbeddingNewParams{
		Model: c.cfg.EmbeddingModel,
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: texts,
		},
	}

	// 如果配置中指定了 EmbeddingDim，则通过 param.NewOpt 设置请求维度
	if c.cfg.EmbeddingDim != 0 {
		params.Dimensions = param.NewOpt(int64(c.cfg.EmbeddingDim))
	}

	resp, err := c.sdk.Embeddings.New(ctx, params)
	if err != nil {
		return nil, err
	}

	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("embeddings: expected %d vectors; got %d", len(texts), len(resp.Data))
	}

	vecs := make([][]float32, len(texts))
	for _, d := range resp.Data {
		idx := int(d.Index)
		if idx < 0 || idx >= len(vecs) {
			return nil, fmt.Errorf("embeddings: index %d out of range", idx)
		}
		vec := make([]float32, len(d.Embedding))
		for i, f := range d.Embedding {
			vec[i] = float32(f)
		}
		vecs[idx] = vec
	}

	return vecs, nil
}

