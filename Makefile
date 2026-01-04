CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

shellgpt.out: shellgpt.c
	$(CC) $(CFLAGS) shellgpt.c $(LDFLAGS) -o shellgpt.out

run: shellgpt.out
	@OPENBLAS_NUM_THREADS=6 ./shellgpt.out 20260103_140047_gpt.bin

clean:
	rm -f shellgpt.out

.PHONY: run clean