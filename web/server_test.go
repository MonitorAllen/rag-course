package web

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"rag-course/llm"
)

func TestWithInlineContextAddsContextToLastUserMessage(t *testing.T) {
	history := []llm.Message{
		{Role: "assistant", Content: "可以。"},
		{Role: "user", Content: "它支持上传吗？"},
	}

	got := withInlineContext(history, "文档上下文")

	if len(got) != len(history) {
		t.Fatalf("len = %d, want %d", len(got), len(history))
	}
	if got[1].Role != "user" {
		t.Fatalf("last role = %q, want user", got[1].Role)
	}
	if !strings.Contains(got[1].Content, "文档上下文") {
		t.Fatalf("last content does not include context: %q", got[1].Content)
	}
	if !strings.Contains(got[1].Content, "它支持上传吗？") {
		t.Fatalf("last content does not include question: %q", got[1].Content)
	}
	if strings.Contains(history[1].Content, "文档上下文") {
		t.Fatal("withInlineContext mutated original history")
	}
}

func TestChatPageRendersComposerControls(t *testing.T) {
	srv, err := New(nil, nil, nil, &Options{Title: "RAG Chat"})
	if err != nil {
		t.Fatalf("new server: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/chat", nil)
	rr := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d", rr.Code)
	}
	body := rr.Body.String()
	for _, want := range []string{
		`id="log"`,
		`id="messageInput"`,
		`id="fileInput"`,
		`id="imageDialogFile"`,
		`id="sendButton"`,
		`Ask a question`,
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("chat page missing %s", want)
		}
	}
}

func TestChatPageSuppressesNativeInputFocusOutline(t *testing.T) {
	srv, err := New(nil, nil, nil, &Options{Title: "RAG Chat"})
	if err != nil {
		t.Fatalf("new server: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/chat", nil)
	rr := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d", rr.Code)
	}
	body := rr.Body.String()
	if !strings.Contains(body, ".message-input:focus-visible") {
		t.Fatal("chat page should suppress native focus-visible outline on the text input")
	}
}

func TestChatPageUsesSeparateUploadEndpointsWithoutAttachmentPromptInjection(t *testing.T) {
	srv, err := New(nil, nil, nil, &Options{Title: "RAG Chat"})
	if err != nil {
		t.Fatalf("new server: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/chat", nil)
	rr := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d", rr.Code)
	}
	body := rr.Body.String()
	for _, want := range []string{`/api/uploads/files`, `/api/uploads/images`} {
		if !strings.Contains(body, want) {
			t.Fatalf("chat page missing upload endpoint %s", want)
		}
	}
	for _, unwanted := range []string{`pendingAttachments`, `buildUserContent`} {
		if strings.Contains(body, unwanted) {
			t.Fatalf("chat page should not keep upload data for chat prompt injection: found %s", unwanted)
		}
	}
}

func TestChatPageUsesImageUploadDialog(t *testing.T) {
	srv, err := New(nil, nil, nil, &Options{Title: "RAG Chat"})
	if err != nil {
		t.Fatalf("new server: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/chat", nil)
	rr := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d", rr.Code)
	}
	body := rr.Body.String()
	for _, want := range []string{
		`id="imageDialog"`,
		`id="imageDialogFile"`,
		`id="imageDescription"`,
		`id="imageDialogSave"`,
		`id="imageDialogCancel"`,
		`Describe what's in the image`,
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("chat page missing image dialog element %s", want)
		}
	}
	if strings.Contains(body, "window.prompt") {
		t.Fatal("chat page should collect image descriptions with the upload dialog, not window.prompt")
	}
}
