package chat

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"rag-course/llm"
	"strings"
	"sync"
	"time"
)

type Options struct {
	SystemPromptFile string
}

func RunREPL(ctx context.Context, client *llm.Client, opts Options) error {
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	history, err := seedHistory(opts.SystemPromptFile)
	if err != nil {
		return err
	}

	fmt.Println("Chat session started, type 'q' to quit.")

	for {
		fmt.Print("\n> ")
		if !in.Scan() {
			if err := in.Err(); err != nil {
				return err
			}
			return nil
		}

		input := strings.TrimSpace(in.Text())
		if input == "" {
			continue
		}

		if strings.EqualFold(input, "q") || strings.EqualFold(input, "/exit") || strings.EqualFold(input, "exit") || strings.EqualFold(input, "quit") {
			fmt.Println("Goodbye.")
			return nil
		}

		history = append(history, llm.Message{Role: "user", Content: input})

		spin := startSpinner("thinking...")
		var stopOnce sync.Once
		reply, err := client.ChatStream(ctx, history, func(s string) {
			stopOnce.Do(spin.Stop)
			fmt.Print(s)
		})

		stopOnce.Do(spin.Stop)
		fmt.Println()

		if err != nil {
			fmt.Println(os.Stderr, "error: ", err)
			history = history[:len(history)-1]
			continue
		}

		history = append(history, reply)

	}
}

type spinner struct {
	stop chan struct{}
	done chan struct{}
	once sync.Once
}

func startSpinner(laber string) *spinner {
	s := &spinner{stop: make(chan struct{}), done: make(chan struct{})}
	go func() {
		defer close(s.done)
		frames := []string{"|", "/", "-", "\\"}
		t := time.NewTicker(80 * time.Millisecond)
		defer t.Stop()
		i := 0
		for {
			select {
			case <-s.stop:
				fmt.Print("\r\033[K]")
				return
			case <-t.C:
				fmt.Printf("\r%s %s", frames[i%len(frames)], laber)
				i++
			}
		}
	}()
	return s
}

func (s *spinner) Stop() {
	s.once.Do(func() {
		close(s.stop)
	})
	<-s.done
}

func seedHistory(path string) ([]llm.Message, error) {
	if path == "" {
		return nil, nil
	}

	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read system prompt: %w", err)
	}

	content := strings.TrimSpace(string(data))
	if content == "" {
		return nil, nil
	}

	return []llm.Message{{Role: "system", Content: content}}, nil
}
