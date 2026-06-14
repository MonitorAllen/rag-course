package rag

import (
	"strings"
	"testing"

	"rag-course/vector"
)

func TestFormatContextIncludesImagePathAndDescription(t *testing.T) {
	got := formatContext([]vector.Result{{
		Document: vector.Document{
			Content: "A course architecture diagram.",
			Metadata: map[string]string{
				"source":     "diagram.png",
				"type":       "image",
				"image_path": "/uploads/images/diagram.png",
			},
		},
		Score: 0.95,
	}})

	for _, want := range []string{"/uploads/images/diagram.png", "A course architecture diagram."} {
		if !strings.Contains(got, want) {
			t.Fatalf("formatted context missing %q: %s", want, got)
		}
	}
}
