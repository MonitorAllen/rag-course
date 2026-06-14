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
)

const maxUploadBytes int64 = 12 << 20

type fileUploadResponse struct {
	Name        string `json:"name"`
	StoredName  string `json:"storedName"`
	Kind        string `json:"kind"`
	ContentType string `json:"contentType"`
	Size        int64  `json:"size"`
	URL         string `json:"url,omitempty"`
	Chunks      int    `json:"chunks,omitempty"`
	Message     string `json:"message"`
}

type jsonError struct {
	Error string `json:"error"`
}

func (s *Server) handleFileUpload(w http.ResponseWriter, r *http.Request) {
	if !s.canIngest() {
		http.Error(w, "ingest is not configured (no vector store)", http.StatusServiceUnavailable)
		return
	}

	data, name, contentType, err := s.readMultipartFile(w, r)
	if err != nil {
		writeUploadError(w, err)
		return
	}

	if !ingest.IsSupported(name) {
		writeJSON(w, http.StatusUnsupportedMediaType, jsonError{Error: "unsupported file type"})
		return
	}

	resp, err := s.saveDocumentUpload(r, name, contentType, data)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, jsonError{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleImageUpload(w http.ResponseWriter, r *http.Request) {
	if !s.canIngest() {
		http.Error(w, "ingest is not configured (no vector store)", http.StatusServiceUnavailable)
		return
	}

	data, name, contentType, err := s.readMultipartFile(w, r)
	if err != nil {
		writeUploadError(w, err)
		return
	}

	if !isImageUpload(name, contentType) {
		writeJSON(w, http.StatusUnsupportedMediaType, jsonError{Error: "unsupported image type"})
		return
	}

	description := strings.TrimSpace(r.FormValue("description"))
	if description == "" {
		writeJSON(w, http.StatusBadRequest, jsonError{Error: "image description is required"})
		return
	}

	resp, err := s.saveImageUpload(r, name, contentType, data, description)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, jsonError{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) canIngest() bool {
	return s.embedder != nil && s.store != nil
}

var errUploadTooLarge = errors.New("upload too large")
var errInvalidMultipart = errors.New("invalid multipart upload")
var errMissingFileField = errors.New("missing file field")
var errEmptyUpload = errors.New("file is empty")

func (s *Server) readMultipartFile(w http.ResponseWriter, r *http.Request) ([]byte, string, string, error) {
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)
	if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
		if strings.Contains(err.Error(), "request body too large") {
			return nil, "", "", errUploadTooLarge
		}
		return nil, "", "", errInvalidMultipart
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		return nil, "", "", errMissingFileField
	}
	defer file.Close()

	data, err := readUpload(file)
	if err != nil {
		return nil, "", "", err
	}
	if len(data) == 0 {
		return nil, "", "", errEmptyUpload
	}

	name := cleanUploadName(header.Filename)
	contentType := detectContentType(header.Header.Get("Content-Type"), data)
	return data, name, contentType, nil
}

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

func (s *Server) saveDocumentUpload(r *http.Request, name, contentType string, data []byte) (fileUploadResponse, error) {
	if s.processedDir == "" {
		return fileUploadResponse{}, errors.New("document upload directory is not configured")
	}
	if err := os.MkdirAll(s.processedDir, 0o755); err != nil {
		return fileUploadResponse{}, fmt.Errorf("create document directory: %w", err)
	}
	storedName, _, err := uniqueStoredName(name)
	if err != nil {
		return fileUploadResponse{}, err
	}
	if err := os.WriteFile(filepath.Join(s.processedDir, storedName), data, 0o644); err != nil {
		return fileUploadResponse{}, fmt.Errorf("save document: %w", err)
	}
	chunks, err := ingest.ProcessContent(r.Context(), storedName, data, ingest.Options{}, s.embedder, s.store)
	if err != nil {
		return fileUploadResponse{}, fmt.Errorf("ingest document: %w", err)
	}

	return fileUploadResponse{
		Name:        name,
		StoredName:  storedName,
		Kind:        "file",
		ContentType: contentType,
		Size:        int64(len(data)),
		Chunks:      chunks,
		Message:     fmt.Sprintf("Processed %s into %d chunks", name, chunks),
	}, nil
}

func (s *Server) saveImageUpload(r *http.Request, name, contentType string, data []byte, description string) (fileUploadResponse, error) {
	if s.imagesDir == "" {
		return fileUploadResponse{}, errors.New("image upload directory is not configured")
	}
	if err := os.MkdirAll(s.imagesDir, 0o755); err != nil {
		return fileUploadResponse{}, fmt.Errorf("create image directory: %w", err)
	}

	storedName, _, err := uniqueStoredName(name)
	if err != nil {
		return fileUploadResponse{}, err
	}
	if err := os.WriteFile(filepath.Join(s.imagesDir, storedName), data, 0o644); err != nil {
		return fileUploadResponse{}, fmt.Errorf("save image: %w", err)
	}

	imageURL := "/uploads/images/" + url.PathEscape(storedName)
	chunks, err := ingest.ProcessText(r.Context(), storedName, description, ingest.Options{}, s.embedder, s.store, map[string]string{
		"type":          "image",
		"kind":          "image",
		"image_path":    imageURL,
		"image_url":     imageURL,
		"original_name": name,
		"content_type":  contentType,
	})
	if err != nil {
		return fileUploadResponse{}, fmt.Errorf("ingest image description: %w", err)
	}

	return fileUploadResponse{
		Name:        name,
		StoredName:  storedName,
		Kind:        "image",
		ContentType: contentType,
		Size:        int64(len(data)),
		URL:         imageURL,
		Chunks:      chunks,
		Message:     fmt.Sprintf("Processed %s description into %d chunks", name, chunks),
	}, nil
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

func writeUploadError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, errUploadTooLarge):
		writeJSON(w, http.StatusRequestEntityTooLarge, jsonError{Error: "file is too large"})
	case errors.Is(err, errMissingFileField):
		writeJSON(w, http.StatusBadRequest, jsonError{Error: "missing file field"})
	case errors.Is(err, errEmptyUpload):
		writeJSON(w, http.StatusBadRequest, jsonError{Error: "file is empty"})
	default:
		writeJSON(w, http.StatusBadRequest, jsonError{Error: "invalid multipart upload"})
	}
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
