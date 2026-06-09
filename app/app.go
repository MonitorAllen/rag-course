package app

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"

	"rag-course/chat"
	"rag-course/config"
	"rag-course/ingest"
	"rag-course/llm"
	"rag-course/rag"
	"rag-course/vector"
	"rag-course/vector/pgvector"
	"rag-course/web"
)

func Run(parent context.Context, cfg config.Config) error {
	logger := log.New(os.Stderr, "[rag] ", log.LstdFlags)

	ctx, cancel := context.WithCancel(parent)
	defer cancel()

	client := llm.New(cfg)

	embedder := llm.NewEmbedderClient(cfg)

	store, err := openStore(ctx, cfg)
	if err != nil {
		return fmt.Errorf("failed to open vector store: %w", err)
	}

	var wg sync.WaitGroup
	if store != nil {
		wg.Go(func() {
			opts := ingest.Options{
				SourceDir:    cfg.IngestDir,
				ProcessedDir: cfg.ProcessedDir,
			}

			if err := ingest.Watch(ctx, opts, embedder, store, logger); err != nil && ctx.Err() == nil {
				logger.Printf("ingest watcher stop: %v", err)
			}
		})
		logger.Printf("watching %s for new documents", cfg.IngestDir)
	}

	if store != nil {
		defer store.Close()
		logger.Printf("vector store ready")
	}

	var retriever *rag.Retriever
	if store != nil {
		// [配置检索增强生成组件]
		// 当向量数据库 (store) 可用时，初始化 RAG 检索器。
		// Retriever 将负责结合当前的对话历史生成检索词，从 store 获取相关切片，格式化后作为上下文提供给 LLM。
		retriever = rag.New(embedder, store, rag.Options{
			TopK:     5,                       // 每次查询返回最相似的前 5 个文档切片
			Rewriter: rag.NewRewriter(client), // 使用专门的 LLM 将多轮对话历史重写为独立的搜索词，提高查询命中率
		})
	}

	if cfg.HTTPAddr != "" {
		srv, err := web.New(client, embedder, retriever, &web.Options{
			Addr:             cfg.HTTPAddr,
			SystemPromptFile: cfg.SystemPromptFile,
			Store:            store,
			ProcessedDir:     cfg.ProcessedDir,
			ImagesDir:        cfg.ImagesDir,
		})
		if err != nil {
			logger.Printf("[web] failed to create server: %v", err)
		} else {
			wg.Go(func() {
				if err := srv.Run(ctx, cfg.HTTPAddr); err != nil && ctx.Err() == nil {
					logger.Printf("[web] server stopped: %v", err)
				}
			})
			logger.Printf("web UI available at http://localhost%s/chat", cfg.HTTPAddr)
		}
	}

	replErr := chat.RunREPL(ctx, client, retriever, chat.Options{
		SystemPromptFile: cfg.SystemPromptFile,
	})

	cancel()
	wg.Wait()
	return replErr
}

func openStore(ctx context.Context, cfg config.Config) (vector.Store, error) {
	if cfg.DatabaseURL == "" {
		return nil, nil
	}
	s, err := pgvector.New(ctx, pgvector.Options{
		DSN:          cfg.DatabaseURL,
		EmbeddingDim: cfg.EmbeddingDim,
	})
	if err != nil {
		return nil, err
	}

	return s, nil
}
