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
	"testing"

	"rag-course/vector"
)

type uploadResponse struct {
	Name       string `json:"name"`
	StoredName string `json:"storedName"`
	Kind       string `json:"kind"`
	Chunks     int    `json:"chunks"`
	URL        string `json:"url"`
	Message    string `json:"message"`
}

func TestUploadFileIngestsSupportedDocument(t *testing.T) {
	docDir := t.TempDir()
	store := &captureStore{}
	srv := &Server{
		processedDir: docDir,
		embedder:     fakeEmbedder{},
		store:        store,
	}

	body, contentType := multipartBody(t, []multipartPart{{
		Field:       "file",
		Filename:    "../notes.md",
		ContentType: "text/markdown",
		Data:        []byte("# RAG\n\nhello"),
	}})

	req := httptest.NewRequest(http.MethodPost, "/api/uploads/files", body)
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
	if resp.Kind != "file" {
		t.Fatalf("kind = %q, want file", resp.Kind)
	}
	if resp.Name != "notes.md" {
		t.Fatalf("name = %q, want sanitized base name", resp.Name)
	}
	if resp.Chunks != 1 {
		t.Fatalf("chunks = %d, want 1", resp.Chunks)
	}
	if resp.Message == "" {
		t.Fatal("success message is empty")
	}
	if _, err := os.Stat(filepath.Join(docDir, resp.StoredName)); err != nil {
		t.Fatalf("stored document not found: %v", err)
	}
	if len(store.docs) != 1 {
		t.Fatalf("upserted docs = %d, want 1", len(store.docs))
	}
	if store.docs[0].Metadata["source"] != resp.StoredName {
		t.Fatalf("doc source = %q, want %q", store.docs[0].Metadata["source"], resp.StoredName)
	}
}

func TestUploadFileRejectsUnsupportedDocumentType(t *testing.T) {
	srv := &Server{
		processedDir: t.TempDir(),
		embedder:     fakeEmbedder{},
		store:        &captureStore{},
	}

	body, contentType := multipartBody(t, []multipartPart{{
		Field:       "file",
		Filename:    "notes.pdf",
		ContentType: "application/pdf",
		Data:        []byte("%PDF"),
	}})

	req := httptest.NewRequest(http.MethodPost, "/api/uploads/files", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()

	srv.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusUnsupportedMediaType {
		t.Fatalf("status = %d, want %d; body = %s", rr.Code, http.StatusUnsupportedMediaType, rr.Body.String())
	}
}

func TestUploadImageRequiresDescription(t *testing.T) {
	srv := &Server{
		imagesDir: t.TempDir(),
		embedder:  fakeEmbedder{},
		store:     &captureStore{},
	}

	body, contentType := multipartBody(t, []multipartPart{{
		Field:       "file",
		Filename:    "diagram.png",
		ContentType: "image/png",
		Data:        pngBytes(),
	}})

	req := httptest.NewRequest(http.MethodPost, "/api/uploads/images", body)
	req.Header.Set("Content-Type", contentType)
	rr := httptest.NewRecorder()

	srv.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d; body = %s", rr.Code, http.StatusBadRequest, rr.Body.String())
	}
}

func TestUploadImageSavesFileAndIngestsDescription(t *testing.T) {
	imageDir := t.TempDir()
	store := &captureStore{}
	srv := &Server{
		imagesDir: imageDir,
		embedder:  fakeEmbedder{},
		store:     store,
	}

	body, contentType := multipartBody(t, []multipartPart{
		{
			Field:       "file",
			Filename:    "../diagram.png",
			ContentType: "image/png",
			Data:        pngBytes(),
		},
		{
			Field: "description",
			Data:  []byte("A course architecture diagram."),
		},
	})

	req := httptest.NewRequest(http.MethodPost, "/api/uploads/images", body)
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
	if resp.Name != "diagram.png" {
		t.Fatalf("name = %q, want sanitized base name", resp.Name)
	}
	if resp.URL == "" {
		t.Fatal("image URL is empty")
	}
	if resp.Chunks != 1 {
		t.Fatalf("chunks = %d, want 1", resp.Chunks)
	}
	if _, err := os.Stat(filepath.Join(imageDir, resp.StoredName)); err != nil {
		t.Fatalf("stored image not found: %v", err)
	}
	if len(store.docs) != 1 {
		t.Fatalf("upserted docs = %d, want 1", len(store.docs))
	}
	doc := store.docs[0]
	if doc.Content != "A course architecture diagram." {
		t.Fatalf("embedded content = %q, want description", doc.Content)
	}
	if doc.Metadata["kind"] != "image" {
		t.Fatalf("metadata kind = %q, want image", doc.Metadata["kind"])
	}
	if doc.Metadata["type"] != "image" {
		t.Fatalf("metadata type = %q, want image", doc.Metadata["type"])
	}
	if doc.Metadata["image_path"] != resp.URL {
		t.Fatalf("metadata image_path = %q, want %q", doc.Metadata["image_path"], resp.URL)
	}
	if doc.Metadata["image_url"] != resp.URL {
		t.Fatalf("metadata image_url = %q, want %q", doc.Metadata["image_url"], resp.URL)
	}
}

func TestUploadedImagesAreServedByFileServer(t *testing.T) {
	imageDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(imageDir, "diagram.png"), pngBytes(), 0o644); err != nil {
		t.Fatalf("write image: %v", err)
	}
	srv := &Server{imagesDir: imageDir}

	req := httptest.NewRequest(http.MethodGet, "/uploads/images/diagram.png", nil)
	rr := httptest.NewRecorder()

	srv.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("served image status = %d", rr.Code)
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

type multipartPart struct {
	Field       string
	Filename    string
	ContentType string
	Data        []byte
}

func multipartBody(t *testing.T, parts []multipartPart) (io.Reader, string) {
	t.Helper()

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	for _, p := range parts {
		header := map[string]string{
			"Content-Disposition": `form-data; name="` + p.Field + `"`,
		}
		if p.Filename != "" {
			header["Content-Disposition"] = `form-data; name="` + p.Field + `"; filename="` + p.Filename + `"`
		}
		if p.ContentType != "" {
			header["Content-Type"] = p.ContentType
		}

		part, err := writer.CreatePart(textprotoMIMEHeader(header))
		if err != nil {
			t.Fatalf("create multipart part: %v", err)
		}
		if _, err := part.Write(p.Data); err != nil {
			t.Fatalf("write multipart part: %v", err)
		}
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

func pngBytes() []byte {
	return []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 'r', 'a', 'g'}
}
