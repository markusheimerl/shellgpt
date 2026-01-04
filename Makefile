CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

all: shellgpt.out trim.out

shellgpt.out: shellgpt.c
	$(CC) $(CFLAGS) shellgpt.c $(LDFLAGS) -o shellgpt.out

trim.out: trim.c
	$(CC) $(CFLAGS) trim.c -o trim.out

trim: trim.out
	@if [ ! -f "20260103_140047_gpt.bin" ]; then \
		echo "Error: 20260103_140047_gpt.bin not found"; \
		exit 1; \
	fi
	@echo "Trimming model file..."
	./trim.out 20260103_140047_gpt.bin
	@echo ""
	@echo "File size comparison:"
	@ls -lh 20260103_140047_gpt.bin 20260103_140047_gpt_trim.bin | awk '{print $$9 " : " $$5}'

run: shellgpt.out
	@if [ ! -f "20260103_140047_gpt_trim.bin" ]; then \
		echo "Trimmed model not found. Run 'make trim' first."; \
		exit 1; \
	fi
	@OPENBLAS_NUM_THREADS=6 ./shellgpt.out 20260103_140047_gpt_trim.bin

clean:
	rm -f shellgpt.out trim.out

.PHONY: all run clean trim