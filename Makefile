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

trim: trim.out
	@./trim.out $$(ls -t *_gpt.bin 2>/dev/null | grep -v "_trim.bin" | head -n1)

run: shellgpt.out decompress
	@./shellgpt.out $$(ls -t *_gpt_trim.bin 2>/dev/null | head -n1)

clean:
	rm -f shellgpt.out trim.out

.PHONY: all run clean trim install uninstall decompress