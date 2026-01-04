CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

all: shellgpt.out trim.out

shellgpt.out: shellgpt.c
	$(CC) $(CFLAGS) shellgpt.c $(LDFLAGS) -o shellgpt.out

trim.out: trim.c
	$(CC) $(CFLAGS) trim.c -o trim.out

install: shellgpt.out
	install -d $(DESTDIR)/usr/local/bin
	install -d $(DESTDIR)/usr/local/share/shellgpt
	install -m 755 shellgpt.out $(DESTDIR)/usr/local/bin/shellgpt
	@MODEL=$$(ls -t *_gpt_trim.bin 2>/dev/null | head -n1); \
	if [ -n "$$MODEL" ]; then \
		install -m 644 $$MODEL $(DESTDIR)/usr/local/share/shellgpt/model.bin; \
		echo "✓ Installed model: $$MODEL -> /usr/local/share/shellgpt/model.bin"; \
	else \
		echo "⚠ Warning: No *_gpt_trim.bin file found"; \
		echo "  Copy a model to /usr/local/share/shellgpt/model.bin manually"; \
		exit 1; \
	fi
	@echo "✓ shellgpt installed successfully!"
	@echo ""
	@echo "Usage: shellgpt \"your question here\""

uninstall:
	rm -f /usr/local/bin/shellgpt
	rm -rf /usr/local/share/shellgpt
	@echo "✓ shellgpt uninstalled"

trim: trim.out
	@./trim.out $$(ls -t *_gpt.bin 2>/dev/null | grep -v "_trim.bin" | head -n1)

run: shellgpt.out
	@OPENBLAS_NUM_THREADS=6 ./shellgpt.out $$(ls -t *_gpt_trim.bin 2>/dev/null | head -n1)

clean:
	rm -f shellgpt.out trim.out

.PHONY: all run clean trim install uninstall