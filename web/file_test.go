package web

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"rag-course/vector"
)

type uploadResponse struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	StoredName  string `json:"storedName"`
	Kind        string `json:"kind"`
	ContentType string `json:"contentType"`
	Size        int64  `json:"size"`
	URL         string `json:"url"`
	Ingested    bool   `json:"ingested"`
	Chunks      int    `json:"chunks"`
}

func TestUploadImageSavesAndServesFile(t *testing.T) {
	imageDir := t.TempDir()
	srv := &Server{images: imageDir}

	body, contentType := multipartBody(t, "file", "diagram.png", "image/png", []byte{
		0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 'r', 'a', 'g',
	})

	req := httptest.NewRequest(http.MethodPost, "/api/uploads", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()

	srv.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("upload status = %d, body = %s", rr.Code, rr.Body.String())
	}

	var resp uploadResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if resp.Kind != "image" {
		t.Fatalf("kind = %q, want image", resp.Kind)
	}
	if resp.URL == "" {
		t.Fatal("image URL is empty")
	}
	if _, err := os.Stat(filepath.Join(imageDir, resp.StoredName)); err != nil {
		t.Fatalf("stored image not found: %v", err)
	}

	getReq := httptest.NewRequest(http.MethodGet, resp.URL, nil)
	getRR := httptest.NewRecorder()
	srv.Routes().ServeHTTP(getRR, getReq)
	if getRR.Code != http.StatusOK {
		t.Fatalf("served image status = %d", getRR.Code)
	}
}

func TestUploadDocumentSavesMarkdownWhenStoreUnavailable(t *testing.T) {
	docDir := t.TempDir()
	srv := &Server{ProcessedDir: docDir}

	body, contentType := multipartBody(t, "file", "../notes.md", "text/markdown", []byte("# RAG\n\nhello"))

	req := httptest.NewRequest(http.MethodPost, "/api/uploads", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()

	srv.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("upload status = %d, body = %s", rr.Code, rr.Body.String())
	}

	var resp uploadResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if resp.Kind != "document" {
		t.Fatalf("kind = %q, want document", resp.Kind)
	}
	if resp.Name != "notes.md" {
		t.Fatalf("name = %q, want sanitized base name", resp.Name)
	}
	if resp.Ingested {
		t.Fatal("document should not be marked ingested when store is unavailable")
	}
	if resp.StoredName == "" {
		t.Fatal("storedName is empty")
	}
	if strings.Contains(resp.StoredName, "..") || strings.ContainsAny(resp.StoredName, `/\`) {
		t.Fatalf("storedName is unsafe: %q", resp.StoredName)
	}
	if _, err := os.Stat(filepath.Join(docDir, resp.StoredName)); err != nil {
		t.Fatalf("stored document not found: %v", err)
	}
}

func TestUploadDocumentIngestsWhenStoreAvailable(t *testing.T) {
	docDir := t.TempDir()
	store := &captureStore{}
	srv := &Server{
		ProcessedDir: docDir,
		embedder:     fakeEmbedder{},
		store:        store,
	}

	body, contentType := multipartBody(t, "file", "notes.md", "text/markdown", []byte("# RAG\n\nhello"))

	req := httptest.NewRequest(http.MethodPost, "/api/uploads", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()

	srv.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("upload status = %d, body = %s", rr.Code, rr.Body.String())
	}

	var resp uploadResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode upload response: %v", err)
	}
	if !resp.Ingested {
		t.Fatal("document should be marked ingested")
	}
	if resp.Chunks != 1 {
		t.Fatalf("chunks = %d, want 1", resp.Chunks)
	}
	if len(store.docs) != 1 {
		t.Fatalf("upserted docs = %d, want 1", len(store.docs))
	}
	if store.deletedSource != resp.StoredName {
		t.Fatalf("deleted source = %q, want %q", store.deletedSource, resp.StoredName)
	}
	if store.docs[0].Metadata["source"] != resp.StoredName {
		t.Fatalf("doc source = %q, want %q", store.docs[0].Metadata["source"], resp.StoredName)
	}
}

type fakeEmbedder struct{}

func (fakeEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	vecs := make([][]float32, len(texts))
	for i := range texts {
		vecs[i] = []float32{1}
	}
	return vecs, nil
}

type captureStore struct {
	docs          []vector.Document
	deletedSource string
}

func (s *captureStore) Upsert(ctx context.Context, docs []vector.Document) error {
	s.docs = append(s.docs, docs...)
	return nil
}

func (s *captureStore) Query(ctx context.Context, embedding []float32, topK int) ([]vector.Result, error) {
	return nil, nil
}

func (s *captureStore) Delete(ctx context.Context, ids []string) error {
	return nil
}

func (s *captureStore) DeleteBySource(ctx context.Context, source string) error {
	s.deletedSource = source
	return nil
}

func (s *captureStore) Close() error {
	return nil
}

func multipartBody(t *testing.T, field, filename, contentType string, data []byte) (io.Reader, string) {
	t.Helper()

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreatePart(textprotoMIMEHeader(map[string]string{
		"Content-Disposition": `form-data; name="` + field + `"; filename="` + filename + `"`,
		"Content-Type":        contentType,
	}))
	if err != nil {
		t.Fatalf("create multipart part: %v", err)
	}
	if _, err := part.Write(data); err != nil {
		t.Fatalf("write multipart part: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	return &body, writer.FormDataContentType()
}

func textprotoMIMEHeader(values map[string]string) textproto.MIMEHeader {
	header := make(textproto.MIMEHeader, len(values))
	for k, v := range values {
		header.Set(k, v)
	}
	return header
}
