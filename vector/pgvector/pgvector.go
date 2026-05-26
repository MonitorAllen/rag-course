package pgvector

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	pgxvec "github.com/pgvector/pgvector-go/pgx"
)

type Options struct {
	DSN          string
	EmbeddingDim int
}

type Store struct {
	pool *pgxpool.Pool
}

func (s *Store) migrate(ctx context.Context, dim int) error {
	stmts := []string{
		fmt.Sprintf(
			`CREATE TABLE IF NOT EXISTS documents (
			id		TEXT PRIMARY KEY,
			content		TEXT NOT NULL,
			metadata	JSONB NOT NULL DEFAULT '{}'::JSONB,
			embedding	VECTOR(%d) NOT NULL
			created_at	TIMESTAMPZ NOT NULL DEFAULT now()
		)`,
			dim,
		),
		"CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING hnsw (embedding vector_cosine_ops)",
	}

	for _, stmt := range stmts {
		if _, err := s.pool.Exec(ctx, stmt); err != nil {
			return fmt.Errorf("exec %q: %w", firstLine(stmt), err)
		}
	}
	return nil

}

func firstLine(s string) string {
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			return s[:i]
		}
	}
	return s
}

func New(ctx context.Context, opts Options) (*Store, error) {
	if opts.DSN == "" {
		return nil, errors.New("pgvector: DSN is required")
	}
	if opts.EmbeddingDim <= 0 {
		return nil, errors.New("pgvector: EmbeddingDim must be positive")
	}

	cfg, err := pgxpool.ParseConfig(opts.DSN)
	if err != nil {
		return nil, fmt.Errorf("pgvector: failed to parse DSN: %w", err)
	}

	if err := ensureExtension(ctx, opts.DSN); err != nil {
		return nil, fmt.Errorf("pgvector: failed to ensure extensions: %w", err)
	}

	cfg.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		return pgxvec.RegisterTypes(ctx, conn)
	}

	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("pgvector: failed to create pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		return nil, fmt.Errorf("pgvector: failed to ping pool: %w", err)
	}

	s := &Store{pool: pool}
	if err := s.migrate(ctx, opts.EmbeddingDim); err != nil {
		pool.Close()
		return nil, fmt.Errorf("pgvector: failed to migrate: %w", err)
	}
	return s, nil
}

func ensureExtension(ctx context.Context, dsn string) error {
	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return err
	}
	defer conn.Close(ctx)

	_, err = conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")
	return err
}
