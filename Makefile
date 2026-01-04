CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

all: shellgpt.out trim.out

shellgpt.out: shellgpt.c
	$(CC) $(CFLAGS) shellgpt.c $(LDFLAGS) -o shellgpt.out

trim.out: trim.c
	$(CC) $(CFLAGS) trim.c -o trim.out

trim: trim.out
	@./trim.out $$(ls -t *_gpt.bin 2>/dev/null | grep -v "_trim.bin" | head -n1)

run: shellgpt.out
	@OPENBLAS_NUM_THREADS=6 ./shellgpt.out $$(ls -t *_gpt_trim.bin 2>/dev/null | head -n1)

clean:
	rm -f shellgpt.out trim.out

.PHONY: all run clean trim