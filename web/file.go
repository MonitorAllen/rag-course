package web

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"rag-course/ingest"
	"strings"
	"unicode"

	"github.com/go-chi/chi/v5"
)

const maxUploadBytes int64 = 12 << 20

type fileUploadResponse struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	StoredName  string `json:"storedName"`
	Kind        string `json:"kind"`
	ContentType string `json:"contentType"`
	Size        int64  `json:"size"`
	URL         string `json:"url,omitempty"`
	Ingested    bool   `json:"ingested"`
	Chunks      int    `json:"chunks,omitempty"`
}

type jsonError struct {
	Error string `json:"error"`
}

func (s *Server) handleUpload(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)
	if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
		status := http.StatusBadRequest
		if strings.Contains(err.Error(), "request body too large") {
			status = http.StatusRequestEntityTooLarge
		}
		writeJSON(w, status, jsonError{Error: "invalid multipart upload"})
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		writeJSON(w, http.StatusBadRequest, jsonError{Error: "missing file field"})
		return
	}
	defer file.Close()

	data, err := readUpload(file)
	if err != nil {
		if errors.Is(err, errUploadTooLarge) {
			writeJSON(w, http.StatusRequestEntityTooLarge, jsonError{Error: "file is too large"})
			return
		}
		writeJSON(w, http.StatusBadRequest, jsonError{Error: "failed to read uploaded file"})
		return
	}
	if len(data) == 0 {
		writeJSON(w, http.StatusBadRequest, jsonError{Error: "file is empty"})
		return
	}

	name := cleanUploadName(header.Filename)
	contentType := detectContentType(header.Header.Get("Content-Type"), data)

	var resp fileUploadResponse
	if isImageUpload(name, contentType) {
		resp, err = s.saveImageUpload(name, contentType, data)
	} else if ingest.IsSupported(name) {
		resp, err = s.saveDocumentUpload(r, name, contentType, data)
	} else {
		writeJSON(w, http.StatusUnsupportedMediaType, jsonError{Error: "unsupported file type"})
		return
	}
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, jsonError{Error: err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleImage(w http.ResponseWriter, r *http.Request) {
	if s.images == "" {
		http.NotFound(w, r)
		return
	}

	name := chi.URLParam(r, "name")
	if !isSafeStoredName(name) {
		http.NotFound(w, r)
		return
	}

	http.ServeFile(w, r, filepath.Join(s.images, name))
}

var errUploadTooLarge = errors.New("upload too large")

func readUpload(r io.Reader) ([]byte, error) {
	limited := io.LimitReader(r, maxUploadBytes+1)
	data, err := io.ReadAll(limited)
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxUploadBytes {
		return nil, errUploadTooLarge
	}
	return data, nil
}

func (s *Server) saveImageUpload(name, contentType string, data []byte) (fileUploadResponse, error) {
	if s.images == "" {
		return fileUploadResponse{}, errors.New("image upload directory is not configured")
	}
	if err := os.MkdirAll(s.images, 0o755); err != nil {
		return fileUploadResponse{}, fmt.Errorf("create image directory: %w", err)
	}

	storedName, id, err := uniqueStoredName(name)
	if err != nil {
		return fileUploadResponse{}, err
	}
	if err := os.WriteFile(filepath.Join(s.images, storedName), data, 0o644); err != nil {
		return fileUploadResponse{}, fmt.Errorf("save image: %w", err)
	}

	return fileUploadResponse{
		ID:          id,
		Name:        name,
		StoredName:  storedName,
		Kind:        "image",
		ContentType: contentType,
		Size:        int64(len(data)),
		URL:         "/uploads/images/" + url.PathEscape(storedName),
	}, nil
}

func (s *Server) saveDocumentUpload(r *http.Request, name, contentType string, data []byte) (fileUploadResponse, error) {
	if s.ProcessedDir == "" {
		return fileUploadResponse{}, errors.New("document upload directory is not configured")
	}
	if err := os.MkdirAll(s.ProcessedDir, 0o755); err != nil {
		return fileUploadResponse{}, fmt.Errorf("create document directory: %w", err)
	}

	storedName, id, err := uniqueStoredName(name)
	if err != nil {
		return fileUploadResponse{}, err
	}
	if err := os.WriteFile(filepath.Join(s.ProcessedDir, storedName), data, 0o644); err != nil {
		return fileUploadResponse{}, fmt.Errorf("save document: %w", err)
	}

	resp := fileUploadResponse{
		ID:          id,
		Name:        name,
		StoredName:  storedName,
		Kind:        "document",
		ContentType: contentType,
		Size:        int64(len(data)),
	}

	if s.embedder != nil && s.store != nil {
		chunks, err := ingest.ProcessContent(r.Context(), storedName, data, ingest.Options{}, s.embedder, s.store)
		if err != nil {
			return fileUploadResponse{}, fmt.Errorf("ingest document: %w", err)
		}
		resp.Ingested = true
		resp.Chunks = chunks
	}

	return resp, nil
}

func cleanUploadName(name string) string {
	base := filepath.Base(strings.TrimSpace(name))
	if base == "." || base == string(filepath.Separator) || base == "" {
		base = "upload"
	}

	ext := strings.ToLower(filepath.Ext(base))
	stem := strings.TrimSuffix(base, filepath.Ext(base))
	var b strings.Builder
	for _, r := range stem {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r):
			b.WriteRune(r)
		case r == '-', r == '_', r == '.', r == ' ':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}

	cleaned := strings.Trim(b.String(), ". -_")
	if cleaned == "" {
		cleaned = "upload"
	}
	if runes := []rune(cleaned); len(runes) > 80 {
		cleaned = string(runes[:80])
	}
	return cleaned + ext
}

func uniqueStoredName(name string) (storedName string, id string, err error) {
	var token [6]byte
	if _, err := rand.Read(token[:]); err != nil {
		return "", "", fmt.Errorf("generate upload id: %w", err)
	}
	id = hex.EncodeToString(token[:])

	ext := strings.ToLower(filepath.Ext(name))
	stem := strings.TrimSuffix(name, filepath.Ext(name))
	storedName = stem + "-" + id + ext
	return storedName, id, nil
}

func detectContentType(header string, data []byte) string {
	if header != "" && header != "application/octet-stream" {
		return strings.ToLower(strings.TrimSpace(header))
	}
	sniff := data
	if len(sniff) > 512 {
		sniff = sniff[:512]
	}
	return strings.ToLower(http.DetectContentType(sniff))
}

func isImageUpload(name, contentType string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".png", ".jpg", ".jpeg", ".gif", ".webp":
	default:
		return false
	}

	switch strings.ToLower(contentType) {
	case "image/png", "image/jpeg", "image/gif", "image/webp":
		return true
	default:
		return strings.HasPrefix(contentType, "image/")
	}
}

func isSafeStoredName(name string) bool {
	if name == "" || filepath.Base(name) != name {
		return false
	}
	return !strings.Contains(name, "..")
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
