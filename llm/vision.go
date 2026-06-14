package llm

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"

	"github.com/openai/openai-go/v3"
)

const captionPrompt = "用 2-3 句话描述该图像以作为搜索索引。 关注可见的主题——它是什么、关键细节、风格。 请勿解释或推测； 仅描述所显示的内容。"

func (c *Client) HasVision() bool {
	return c.cfg.VisionModel != ""
}

func (c *Client) DescribeImage(ctx context.Context, mime string, image []byte) (string, error) {
	if !c.HasVision() {
		return "", errors.New("no vision model configured ")
	}
	if len(image) == 0 {
		return "", errors.New("empty image")
	}

	if mime == "" {
		mime = http.DetectContentType(image)
	}

	dataURL := fmt.Sprintf("data:%s;base64,%s", mime, base64.StdEncoding.EncodeToString(image))

	resp, err := c.sdk.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: c.cfg.VisionModel,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
				openai.TextContentPart(captionPrompt),
				openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
					URL: dataURL,
				}),
			}),
		},
	})
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", errors.New("no choices found")
	}
	return resp.Choices[0].Message.Content, nil
}
