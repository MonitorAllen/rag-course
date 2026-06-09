package web

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io/fs"
	"log"
	"net/http"
	"os"
	"rag-course/llm"
	"rag-course/rag"
	"rag-course/vector"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

//go:embed templates/*.gohtml
var templatesFS embed.FS

type Options struct {
	Addr             string
	SystemPromptFile string
	Title            string
	Store            vector.Store
	ProcessedDir     string
	ImagesDir        string
}

type Server struct {
	client       *llm.Client
	embedder     llm.Embedder
	retriever    *rag.Retriever
	store        vector.Store
	ProcessedDir string
	images       string
	tpl          *template.Template
	system       string
	title        string
}

func New(client *llm.Client, embedder llm.Embedder, retriever *rag.Retriever, opts *Options) (*Server, error) {
	if opts == nil {
		opts = &Options{}
	}

	tpl, err := template.ParseFS(templatesFS, "templates/*.gohtml")
	if err != nil {
		return nil, fmt.Errorf("parse templates: %w", err)
	}

	title := opts.Title
	if title == "" {
		title = "RAG Chat"
	}

	return &Server{
		client:       client,
		embedder:     embedder,
		retriever:    retriever,
		store:        opts.Store,
		ProcessedDir: opts.ProcessedDir,
		images:       opts.ImagesDir,
		tpl:          tpl,
		system:       readSystemPrompt(opts.SystemPromptFile),
		title:        title,
	}, nil
}

func (s *Server) Routes() http.Handler {
	r := chi.NewRouter()
	r.Use(middleware.Recoverer)
	r.Get("/chat", s.handleChatPage)
	r.Post("/api/chat/stream", s.handleChatStream)
	r.Post("/api/uploads", s.handleUpload)
	r.Get("/uploads/images/{name}", s.handleImage)
	return r
}

type chatRequest struct {
	Messages []llm.Message `json:"messages"`
}

func (s *Server) handleChatStream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "server does not support streaming", http.StatusInternalServerError)
		return
	}

	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if len(req.Messages) == 0 {
		http.Error(w, "messages must not be empty", http.StatusBadRequest)
		return
	}

	// 验证最后一条消息是 user 角色（新输入），并校验内容不为空。
	last := req.Messages[len(req.Messages)-1]
	if last.Role != "user" || strings.TrimSpace(last.Content) == "" {
		http.Error(w, "last message must be from user and not empty", http.StatusBadRequest)
		return
	}

	// 准备基础的对话历史和上下文
	ctx := r.Context()
	history := req.Messages
	if s.system != "" {
		history = append([]llm.Message{{Role: "system", Content: s.system}}, history...)
	}

	var contextText string
	if s.retriever != nil {
		var err error
		contextText, err = s.retriever.Retrieve(ctx, history)
		if err != nil {
			log.Printf("[web] retrieve error: %v", err)
		}
	}

	turn := history
	if contextText != "" {
		turn = withInlineContext(history, contextText)
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	send := func(event string, data string) {
		if event != "" {
			fmt.Fprintf(w, "event: %s\n", event)
		}
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	_, err := s.client.ChatStream(r.Context(), turn, func(delta string) {
		enc, _ := json.Marshal(delta)
		send("delta", string(enc))
	})

	if err != nil {
		enc, _ := json.Marshal(err.Error())
		log.Printf("[web] chat error: %v", err)
		// 尝试发送错误消息，但不要阻断连接
		send("error", string(enc))
		return
	}
	send("done", `""`)

}

func withInlineContext(history []llm.Message, contextText string) []llm.Message {
	if len(history) == 0 || contextText == "" {
		return history
	}

	last := history[len(history)-1]
	if last.Role != "user" {
		return history
	}
	out := make([]llm.Message, len(history))
	copy(out, history)
	out[len(out)-1] = llm.Message{
		Role:    "user",
		Content: contextText + "\n\n--- 原问题 ---\n\n" + last.Content,
	}
	return out
}

func (s *Server) handleChatPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := s.tpl.ExecuteTemplate(w, "chat.gohtml", map[string]any{
		"Title": s.title,
	}); err != nil {
		log.Printf("[web] template error: %v", err)
	}
}

func (s *Server) Run(ctx context.Context, addr string) error {
	srv := &http.Server{
		Addr:              addr,
		Handler:           s.Routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		err := srv.ListenAndServe()
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
		errCh <- nil
	}()

	select {
	case <-ctx.Done():
		log.Printf("[web] shutting down")
		// 创建一个带有超时的上下文，用于在服务器关闭时等待
		shutDownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutDownCtx)
		return nil
	case err := <-errCh:
		return err
	}
}

func readSystemPrompt(path string) string {
	if path == "" {
		return ""
	}

	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		fmt.Printf("warning: system prompt %q does not exist\n", path)
		return ""
	}
	if err != nil {
		fmt.Printf("warning: failed to read system prompt %q: %v\n", path, err)
		return ""
	}
	return strings.TrimSpace(string(data))
}
