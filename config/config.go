package config

import (
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

type Config struct {
	BaseURL          string
	APIKey           string
	Model            string
	SystemPromptFile string

	DatabaseURL  string
	EmbeddingDim int

	EmbeddingBaseURL string
	EmbeddingAPIKey  string
	EmbeddingModel   string

	IngestDir    string
	ProcessedDir string
	HTTPAddr     string
	ImagesDir    string
	VisionModel  string
}

func Load() Config {
	_ = godotenv.Load()

	cfg := Config{
		BaseURL:          os.Getenv("OPENAI_BASE_URL"),
		APIKey:           os.Getenv("OPENAI_API_KEY"),
		Model:            os.Getenv("OPENAI_MODEL"),
		SystemPromptFile: os.Getenv("SYSTEM_PROMPT_FILE"),
		DatabaseURL:      os.Getenv("DATABASE_URL"),
		EmbeddingDim:     atoiOr(os.Getenv("EMBEDDING_DIM"), 0),
		EmbeddingBaseURL: os.Getenv("EMBEDDING_BASE_URL"),
		EmbeddingAPIKey:  os.Getenv("EMBEDDING_API_KEY"),
		EmbeddingModel:   os.Getenv("EMBEDDING_MODEL"),
		IngestDir:        os.Getenv("INGEST_DIR"),
		ProcessedDir:     os.Getenv("PROCESSED_DIR"),
		HTTPAddr:         os.Getenv("HTTP_ADDR"),
		ImagesDir:        os.Getenv("IMAGES_DIR"),
		VisionModel:      os.Getenv("VISION_MODEL"),
	}

	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}

	if cfg.Model == "" {
		cfg.Model = "gpt-4o-mini"
	}

	if cfg.EmbeddingDim == 0 {
		cfg.EmbeddingDim = 768
	}

	if cfg.EmbeddingBaseURL == "" {
		cfg.EmbeddingBaseURL = cfg.BaseURL
		if cfg.EmbeddingAPIKey == "" {
			cfg.EmbeddingAPIKey = cfg.APIKey
		}
	}

	if cfg.EmbeddingModel == "" {
		cfg.EmbeddingModel = "text-embedding-3-small"
	}

	if cfg.IngestDir == "" {
		cfg.IngestDir = "./documents"
	}

	if cfg.ProcessedDir == "" {
		cfg.ProcessedDir = "./documents/processed"
	}

	if cfg.HTTPAddr == "" {
		cfg.HTTPAddr = ":8080"
	}

	if cfg.ImagesDir == "" {
		cfg.ImagesDir = "./documents/images"
	}

	if cfg.VisionModel == "" {
		cfg.VisionModel = "gpt-4o-mini"
	}

	return cfg
}

func atoiOr(s string, fallback int) int {
	if s == "" {
		return fallback
	}

	n, err := strconv.Atoi(s)
	if err != nil {
		return fallback
	}
	return n
}
