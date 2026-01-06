CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -static -lopenblas -lm

all: shellgpt.out trim.out

shellgpt.out: shellgpt.c
	$(CC) $(CFLAGS) shellgpt.c $(LDFLAGS) -o shellgpt.out

trim.out: trim.c
	$(CC) $(CFLAGS) trim.c -o trim.out

decompress:
	@if [ ! -f *_gpt_trim.bin ] && [ -f *_gpt_trim.bin.gz ]; then \
		echo "Decompressing model..."; \
		gunzip -k *_gpt_trim.bin.gz; \
	fi

install: shellgpt.out decompress
	install -d $(DESTDIR)/usr/bin
	install -d $(DESTDIR)/usr/share/shellgpt
	install -m 755 shellgpt.out $(DESTDIR)/usr/bin/shellgpt
	@MODEL=$$(ls -t *_gpt_trim.bin 2>/dev/null | head -n1); \
	if [ -n "$$MODEL" ]; then \
		install -m 644 $$MODEL $(DESTDIR)/usr/share/shellgpt/model.bin; \
		echo "✓ Found model: $$MODEL"; \
	else \
		echo "⚠ Warning: No *_gpt_trim.bin file found"; \
		exit 1; \
	fi

uninstall:
	rm -f /usr/bin/shellgpt
	rm -rf /usr/share/shellgpt
	@echo "✓ shellgpt uninstalled"

trim: trim.out
	@./trim.out $$(ls -t *_gpt.bin 2>/dev/null | grep -v "_trim.bin" | head -n1)

run: shellgpt.out decompress
	@./shellgpt.out $$(ls -t *_gpt_trim.bin 2>/dev/null | head -n1)

clean:
	rm -f shellgpt.out trim.out

.PHONY: all run clean trim install uninstall decompress