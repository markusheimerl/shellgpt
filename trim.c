#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// ============================================================================
// Trim Functions - Remove optimizer states from model file
// ============================================================================

void trim_mlp(FILE* fin, FILE* fout) {
    int input_dim, hidden_dim, output_dim;
    fread(&input_dim, sizeof(int), 1, fin);
    fread(&hidden_dim, sizeof(int), 1, fin);
    fread(&output_dim, sizeof(int), 1, fin);
    
    fwrite(&input_dim, sizeof(int), 1, fout);
    fwrite(&hidden_dim, sizeof(int), 1, fout);
    fwrite(&output_dim, sizeof(int), 1, fout);
    
    size_t w1_size = input_dim * hidden_dim;
    size_t w2_size = hidden_dim * output_dim;
    
    float* W1 = malloc(w1_size * sizeof(float));
    float* W2 = malloc(w2_size * sizeof(float));
    
    fread(W1, sizeof(float), w1_size, fin);
    fread(W2, sizeof(float), w2_size, fin);
    
    fwrite(W1, sizeof(float), w1_size, fout);
    fwrite(W2, sizeof(float), w2_size, fout);
    
    // Skip optimizer state (not written to output)
    int t;
    fread(&t, sizeof(int), 1, fin);
    fseek(fin, (w1_size + w1_size + w2_size + w2_size) * sizeof(float), SEEK_CUR);
    
    free(W1);
    free(W2);
}

void trim_attention(FILE* fin, FILE* fout) {
    int d_model;
    bool is_causal, use_rope;
    fread(&d_model, sizeof(int), 1, fin);
    fread(&is_causal, sizeof(bool), 1, fin);
    fread(&use_rope, sizeof(bool), 1, fin);
    
    fwrite(&d_model, sizeof(int), 1, fout);
    fwrite(&is_causal, sizeof(bool), 1, fout);
    fwrite(&use_rope, sizeof(bool), 1, fout);
    
    size_t w_size = d_model * d_model;
    
    float* W_q = malloc(w_size * sizeof(float));
    float* W_k = malloc(w_size * sizeof(float));
    float* W_v = malloc(w_size * sizeof(float));
    float* W_o = malloc(w_size * sizeof(float));
    
    fread(W_q, sizeof(float), w_size, fin);
    fread(W_k, sizeof(float), w_size, fin);
    fread(W_v, sizeof(float), w_size, fin);
    fread(W_o, sizeof(float), w_size, fin);
    
    fwrite(W_q, sizeof(float), w_size, fout);
    fwrite(W_k, sizeof(float), w_size, fout);
    fwrite(W_v, sizeof(float), w_size, fout);
    fwrite(W_o, sizeof(float), w_size, fout);
    
    // Skip optimizer state (not written to output)
    int t;
    fread(&t, sizeof(int), 1, fin);
    fseek(fin, 8 * w_size * sizeof(float), SEEK_CUR);
    
    free(W_q);
    free(W_k);
    free(W_v);
    free(W_o);
}

void trim_transformer(FILE* fin, FILE* fout, int num_layers) {
    for (int i = 0; i < num_layers; i++) {
        printf("  Trimming layer %d/%d...\n", i + 1, num_layers);
        trim_attention(fin, fout);
        trim_mlp(fin, fout);
    }
}

int trim_model(const char* input_file, const char* output_file) {
    FILE* fin = fopen(input_file, "rb");
    if (!fin) {
        fprintf(stderr, "Error opening input file: %s\n", input_file);
        return 1;
    }
    
    FILE* fout = fopen(output_file, "wb");
    if (!fout) {
        fprintf(stderr, "Error opening output file: %s\n", output_file);
        fclose(fin);
        return 1;
    }
    
    printf("Trimming model: %s\n", input_file);
    
    // Read and write GPT header
    int d_model, hidden_dim, num_layers, vocab_size;
    fread(&d_model, sizeof(int), 1, fin);
    fread(&hidden_dim, sizeof(int), 1, fin);
    fread(&num_layers, sizeof(int), 1, fin);
    fread(&vocab_size, sizeof(int), 1, fin);
    
    fwrite(&d_model, sizeof(int), 1, fout);
    fwrite(&hidden_dim, sizeof(int), 1, fout);
    fwrite(&num_layers, sizeof(int), 1, fout);
    fwrite(&vocab_size, sizeof(int), 1, fout);
    
    printf("Model config: d_model=%d, hidden_dim=%d, num_layers=%d, vocab_size=%d\n",
           d_model, hidden_dim, num_layers, vocab_size);
    
    // Token embedding
    printf("Trimming token embeddings...\n");
    size_t emb_size = (size_t)vocab_size * d_model;
    float* token_embedding = malloc(emb_size * sizeof(float));
    
    fread(token_embedding, sizeof(float), emb_size, fin);
    fwrite(token_embedding, sizeof(float), emb_size, fout);
    
    // Skip embedding optimizer state (not written to output)
    int t;
    fread(&t, sizeof(int), 1, fin);
    fseek(fin, 2 * emb_size * sizeof(float), SEEK_CUR);
    
    free(token_embedding);
    
    // Transformer header
    int tf_d_model, tf_hidden_dim, tf_num_layers;
    bool is_causal, use_rope;
    fread(&tf_d_model, sizeof(int), 1, fin);
    fread(&tf_hidden_dim, sizeof(int), 1, fin);
    fread(&tf_num_layers, sizeof(int), 1, fin);
    fread(&is_causal, sizeof(bool), 1, fin);
    fread(&use_rope, sizeof(bool), 1, fin);
    
    fwrite(&tf_d_model, sizeof(int), 1, fout);
    fwrite(&tf_hidden_dim, sizeof(int), 1, fout);
    fwrite(&tf_num_layers, sizeof(int), 1, fout);
    fwrite(&is_causal, sizeof(bool), 1, fout);
    fwrite(&use_rope, sizeof(bool), 1, fout);
    
    // Transformer layers
    printf("Trimming transformer layers...\n");
    trim_transformer(fin, fout, num_layers);
    
    fclose(fin);
    fclose(fout);
    
    printf("\n✓ Successfully trimmed model saved to: %s\n", output_file);
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model_file.bin>\n", argv[0]);
        fprintf(stderr, "Output will be saved as <model_file>_trim.bin\n");
        return 1;
    }
    
    const char* input_file = argv[1];
    
    // Create output filename
    char output_file[1024];
    const char* ext = strrchr(input_file, '.');
    if (ext && strcmp(ext, ".bin") == 0) {
        size_t base_len = ext - input_file;
        snprintf(output_file, sizeof(output_file), "%.*s_trim.bin", (int)base_len, input_file);
    } else {
        snprintf(output_file, sizeof(output_file), "%s_trim.bin", input_file);
    }
    
    return trim_model(input_file, output_file);
}