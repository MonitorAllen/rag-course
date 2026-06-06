// 定义 ingest 包，包含文档导入、分块和监视源目录变动等核心逻辑
package ingest

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"rag-course/llm"
	"rag-course/vector"

	"github.com/fsnotify/fsnotify"
)

// 定义防抖（Debounce）延迟时间为 500 毫秒，防止在文件写入未完成时频繁触发处理逻辑
const debounceDelay = 500 * time.Millisecond

// Watch 启动一个阻塞的文件监视器，监听源目录的变动，提取新文件内容进行向量化并存入向量库中
func Watch(ctx context.Context, opts Options, embedder llm.Embedder, store vector.Store, logger *log.Logger) error {
	// 校验：源目录与已处理目录不能为同一个目录，防止产生处理与移动的死循环
	if filepath.Clean(opts.SourceDir) == filepath.Clean(opts.ProcessedDir) {
		return errors.New("source and processed directories must be different")
	}

	// 创建源目录（权限为 0755），如果目录已存在则什么都不做
	if err := os.MkdirAll(opts.SourceDir, 0o755); err != nil {
		return fmt.Errorf("create source dir: %w", err)
	}

	// 创建已处理目录（权限为 0755），如果目录已存在则什么都不做
	if err := os.MkdirAll(opts.ProcessedDir, 0o755); err != nil {
		return fmt.Errorf("create processed dir: %w", err)
	}

	// 校验：源目录路径清理后不能为空
	if filepath.Clean(opts.SourceDir) == "" {
		return errors.New("source directory must be non-empty")
	}

	// 校验：嵌入模型对象（Embedder）不能为空
	if embedder == nil {
		return errors.New("embedder is required")
	}

	// 校验：向量存储库对象（Store）不能为空
	if store == nil {
		return errors.New("vector store is required")
	}

	// 如果未指定日志记录器，则使用 Go 语言自带的默认日志输出
	if logger == nil {
		logger = log.Default()
	}

	// 创建一个新的 fsnotify 监听器示例
	w, err := fsnotify.NewWatcher()
	if err != nil {
		return err
	}
	// 在 Watch 函数退出时，确保关闭监听器，释放操作系统的文件描述符资源
	defer w.Close()

	// 将源目录添加到监听器的监视列表里，开始监视该目录下的文件变动
	if err := w.Add(opts.SourceDir); err != nil {
		return fmt.Errorf("watch source dir: %w", err)
	}

	// 打印日志，声明文件监视服务已在源目录启动
	logger.Printf("Watch started for %q", opts.SourceDir)

	// 获取已处理目录的绝对路径，用于后续在 shouldProcess 方法中判断是否应该处理该路径下的文件
	processedAbs, err := filepath.Abs(opts.ProcessedDir)
	if err != nil {
		return fmt.Errorf("absolute path of processed dir: %w", err)
	}

	// 定义文件处理逻辑闭包，封装了对单一文件的“提取 -> 向量化 -> 导入向量库 -> 移至已处理目录”的完整工作流
	handle := func(path string) {
		// 调用 ingest 包内的 processOne 函数：读取文件，切块，调用 Embed 转换，最后 Upsert 到 store
		if err := processOne(ctx, path, opts, embedder, store); err != nil {
			// 处理失败时记录错误日志，并提前退出，不移动该文件
			logger.Printf("Failed to process %s: %v", filepath.Base(path), err)
			return
		}

		// 拼接该文件在已处理目录中的目标路径
		dst := filepath.Join(opts.ProcessedDir, filepath.Base(path))
		// 将处理完成的文件从源目录移动到已处理目录中
		if err := os.Rename(path, dst); err != nil {
			// 移动失败时仅记录错误日志，防止阻塞其他流程
			logger.Printf("Failed to move %s to processed: %v", filepath.Base(path), err)
		}

		// 打印成功处理并移动的日志
		logger.Printf("✅ Successfully processed and moved: %s", filepath.Base(path))
	}

	// 读取源目录下的全部现有文件，防止在服务启动前目录里已经存在未处理的文档
	entries, err := os.ReadDir(opts.SourceDir)
	if err != nil {
		return fmt.Errorf("failed to read source dir: %w", err)
	}
	// 遍历源目录下的已有条目
	for _, e := range entries {
		// 忽略子目录，只处理文件
		if e.IsDir() {
			continue
		}
		// 获取文件的完整路径
		p := filepath.Join(opts.SourceDir, e.Name())
		// 为每个已有文件启动一个独立的 goroutine 并发处理
		go handle(p)
	}

	// 声明防抖计时器相关的变量
	var (
		// 互斥锁，用于保护共享的计时器 map 防止并发写操作冲突
		timersMu sync.Mutex
		// 记录每个文件对应的防抖定时器
		timers = make(map[string]*time.Timer)
	)

	// 定义防抖调度闭包。用于在文件被频繁写入或创建时，延迟并合并多次事件触发
	schedule := func(path string) {
		// 加锁，保护 timers map
		timersMu.Lock()
		defer timersMu.Unlock()

		// 如果该文件已经存在一个待执行的定时器
		if t, ok := timers[path]; ok {
			// 重置该定时器，使其重新等待 debounceDelay (500ms)
			t.Reset(debounceDelay)
			return
		}
		// 如果是首次触发，创建一个在延时 500ms 后执行的定时器
		timers[path] = time.AfterFunc(debounceDelay, func() {
			// 定时器触发时，重新加锁，将该文件的定时器记录从 map 中移除
			timersMu.Lock()
			delete(timers, path)
			timersMu.Unlock()
			// 执行文件提取和导入流程
			handle(path)
		})
	}

	// 进入死循环，等待系统信号和文件事件
	for {
		select {
		// 监听上下文取消信号，以便可以随着应用的主进程优雅退出
		case <-ctx.Done():
			// 返回上下文对应的取消错误（即 context.Canceled）
			return ctx.Err()
		// 监听文件监视器的事件通道
		case ev, ok := <-w.Events:
			// 如果通道已被关闭，说明 Watcher 被停止，直接退出循环
			if !ok {
				return nil
			}
			// 仅关注文件的写入（Write）和创建（Create）事件，忽略其他事件（如重命名、删除、修改权限等）
			if ev.Op&(fsnotify.Write|fsnotify.Create) == 0 {
				continue
			}
			// 过滤检查：判断文件路径是否满足处理要求（如是否是隐藏文件、是否在已处理目录中等）
			if !shouldProcess(ev.Name, processedAbs) {
				continue
			}
			// 将合规的事件文件调度到防抖机制中处理
			schedule(ev.Name)
		// 监听文件监视器底层的错误通道
		case err, ok := <-w.Errors:
			// 如果通道已被关闭，直接退出循环
			if !ok {
				return nil
			}
			// 记录监听器发生的底层系统级错误
			logger.Printf("Watcher error: %v", err)
		}
	}
}

// shouldProcess 用于在预处理和监听事件时，过滤掉不需要处理的文件或路径
func shouldProcess(path, processedAbs string) bool {
	// 如果文件名是以 "." 开头的隐藏文件（例如 macOS 的 .DS_Store 或临时文件），则不予处理
	if strings.HasPrefix(filepath.Base(path), ".") {
		return false
	}
	// 获取文件的状态信息，如果文件不存在或者发生其他 IO 错误，直接过滤掉
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return false
	}
	// 获取当前文件的绝对路径，便于安全比较
	abs, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	// 安全校验：如果当前处理的文件绝对路径，以已处理目录的绝对路径为前缀，说明它已在 processed 目录中，不重复处理
	if processedAbs != "" && strings.HasPrefix(abs, processedAbs+string(filepath.Separator)) {
		return false
	}
	// 通过所有检查，允许被处理
	return true
}
