#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <cblas.h>

// ============================================================================
// Structure Definitions
// ============================================================================

typedef struct {
    float* W1; float* W2;
    float* h;      // [hidden_dim] pre-activation
    float* s;      // [hidden_dim] post-swish
    float* output; // [output_dim]
    int input_dim; int hidden_dim; int output_dim;
} MLP;

typedef struct {
    float* W_q; float* W_k; float* W_v; float* W_o;
    float* q; float* k; float* v;   // [d_model] current token projections
    float* z;                        // [d_model] concatenated heads
    float* output;                   // [d_model]
    float* scores; float* probs;     // [seq_len] for attention
    float* K_cache; float* V_cache;  // [seq_len x d_model] KV cache
    int seq_len; int d_model; int num_heads; int head_dim;
    float scale; bool is_causal; bool use_rope;
} Attention;

typedef struct {
    Attention** attention_layers;
    MLP** mlp_layers;
    float* norm1; float* norm2;  // [d_model] RMSNorm outputs
    int seq_len; int d_model; int hidden_dim; int num_layers;
} Transformer;

typedef struct {
    float* token_embedding;
    Transformer* transformer;
    float* x;          // [d_model] current hidden state
    float* final_norm; // [d_model]
    int seq_len; int d_model; int hidden_dim; int num_layers; int vocab_size;
} GPT;

// ============================================================================
// MLP Functions
// ============================================================================

MLP* init_mlp(int input_dim, int hidden_dim, int output_dim) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    mlp->input_dim = input_dim;
    mlp->hidden_dim = hidden_dim;
    mlp->output_dim = output_dim;
    mlp->W1 = (float*)malloc(input_dim * hidden_dim * sizeof(float));
    mlp->W2 = (float*)malloc(hidden_dim * output_dim * sizeof(float));
    mlp->h = (float*)malloc(hidden_dim * sizeof(float));
    mlp->s = (float*)malloc(hidden_dim * sizeof(float));
    mlp->output = (float*)malloc(output_dim * sizeof(float));
    return mlp;
}

void free_mlp(MLP* mlp) {
    free(mlp->W1); free(mlp->W2);
    free(mlp->h); free(mlp->s); free(mlp->output);
    free(mlp);
}

void forward_mlp(MLP* mlp, float* x) {
    // h = x @ W1
    cblas_sgemv(CblasRowMajor, CblasTrans, mlp->input_dim, mlp->hidden_dim,
                1.0f, mlp->W1, mlp->hidden_dim, x, 1, 0.0f, mlp->h, 1);
    // Swish: s = h * sigmoid(h)
    for (int i = 0; i < mlp->hidden_dim; i++)
        mlp->s[i] = mlp->h[i] / (1.0f + expf(-mlp->h[i]));
    // output = s @ W2
    cblas_sgemv(CblasRowMajor, CblasTrans, mlp->hidden_dim, mlp->output_dim,
                1.0f, mlp->W2, mlp->output_dim, mlp->s, 1, 0.0f, mlp->output, 1);
}

MLP* deserialize_mlp(FILE* f) {
    int input_dim, hidden_dim, output_dim;
    fread(&input_dim, sizeof(int), 1, f);
    fread(&hidden_dim, sizeof(int), 1, f);
    fread(&output_dim, sizeof(int), 1, f);
    
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim);
    size_t w1_size = input_dim * hidden_dim;
    size_t w2_size = hidden_dim * output_dim;
    
    fread(mlp->W1, sizeof(float), w1_size, f);
    fread(mlp->W2, sizeof(float), w2_size, f);
    
    // Skip optimizer state
    int t; fread(&t, sizeof(int), 1, f);
    fseek(f, (w1_size + w1_size + w2_size + w2_size) * sizeof(float), SEEK_CUR);
    
    return mlp;
}

// ============================================================================
// Attention Functions
// ============================================================================

Attention* init_attention(int seq_len, int d_model, int num_heads, bool is_causal, bool use_rope) {
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    attn->seq_len = seq_len;
    attn->d_model = d_model;
    attn->num_heads = num_heads;
    attn->head_dim = d_model / num_heads;
    attn->scale = 1.0f / sqrtf(attn->head_dim);
    attn->is_causal = is_causal;
    attn->use_rope = use_rope;
    
    size_t w_size = d_model * d_model;
    attn->W_q = (float*)malloc(w_size * sizeof(float));
    attn->W_k = (float*)malloc(w_size * sizeof(float));
    attn->W_v = (float*)malloc(w_size * sizeof(float));
    attn->W_o = (float*)malloc(w_size * sizeof(float));
    
    attn->q = (float*)malloc(d_model * sizeof(float));
    attn->k = (float*)malloc(d_model * sizeof(float));
    attn->v = (float*)malloc(d_model * sizeof(float));
    attn->z = (float*)malloc(d_model * sizeof(float));
    attn->output = (float*)malloc(d_model * sizeof(float));
    attn->scores = (float*)malloc(seq_len * sizeof(float));
    attn->probs = (float*)malloc(seq_len * sizeof(float));
    
    attn->K_cache = (float*)malloc((size_t)seq_len * d_model * sizeof(float));
    attn->V_cache = (float*)malloc((size_t)seq_len * d_model * sizeof(float));
    
    return attn;
}

void free_attention(Attention* attn) {
    free(attn->W_q); free(attn->W_k); free(attn->W_v); free(attn->W_o);
    free(attn->q); free(attn->k); free(attn->v); free(attn->z);
    free(attn->output); free(attn->scores); free(attn->probs);
    free(attn->K_cache); free(attn->V_cache);
    free(attn);
}

static void rope_apply(float* q, float* k, int pos, int d_model) {
    for (int d_pair = 0; d_pair < d_model / 2; d_pair++) {
        int d = d_pair * 2;
        float theta = powf(10000.0f, -((float)d / (float)d_model));
        float angle = (float)pos * theta;
        float cos_a = cosf(angle), sin_a = sinf(angle);
        
        float q0 = q[d], q1 = q[d + 1];
        q[d] = q0 * cos_a - q1 * sin_a;
        q[d + 1] = q0 * sin_a + q1 * cos_a;
        
        float k0 = k[d], k1 = k[d + 1];
        k[d] = k0 * cos_a - k1 * sin_a;
        k[d + 1] = k0 * sin_a + k1 * cos_a;
    }
}

void forward_attention(Attention* attn, float* x, int pos) {
    int d = attn->d_model;
    int hd = attn->head_dim;
    int nh = attn->num_heads;
    
    // q, k, v = x @ W_q, x @ W_k, x @ W_v
    cblas_sgemv(CblasRowMajor, CblasTrans, d, d, 1.0f, attn->W_q, d, x, 1, 0.0f, attn->q, 1);
    cblas_sgemv(CblasRowMajor, CblasTrans, d, d, 1.0f, attn->W_k, d, x, 1, 0.0f, attn->k, 1);
    cblas_sgemv(CblasRowMajor, CblasTrans, d, d, 1.0f, attn->W_v, d, x, 1, 0.0f, attn->v, 1);
    
    // Apply RoPE
    if (attn->use_rope) rope_apply(attn->q, attn->k, pos, d);
    
    // Store k, v in cache
    memcpy(attn->K_cache + pos * d, attn->k, d * sizeof(float));
    memcpy(attn->V_cache + pos * d, attn->v, d * sizeof(float));
    
    // Multi-head attention with cache
    memset(attn->z, 0, d * sizeof(float));
    
    for (int h = 0; h < nh; h++) {
        float* q_h = attn->q + h * hd;
        float* z_h = attn->z + h * hd;
        
        // Compute scores for this head
        float max_score = -1e30f;
        for (int j = 0; j <= pos; j++) {
            float* K_j_h = attn->K_cache + j * d + h * hd;
            float score = cblas_sdot(hd, q_h, 1, K_j_h, 1) * attn->scale;
            attn->scores[j] = score;
            if (score > max_score) max_score = score;
        }
        
        // Softmax
        float sum_exp = 0.0f;
        for (int j = 0; j <= pos; j++) {
            attn->probs[j] = expf(attn->scores[j] - max_score);
            sum_exp += attn->probs[j];
        }
        float inv_sum = 1.0f / sum_exp;
        
        // Weighted sum of values
        for (int j = 0; j <= pos; j++) {
            float w = attn->probs[j] * inv_sum;
            float* V_j_h = attn->V_cache + j * d + h * hd;
            cblas_saxpy(hd, w, V_j_h, 1, z_h, 1);
        }
    }
    
    // Output projection
    cblas_sgemv(CblasRowMajor, CblasTrans, d, d, 1.0f, attn->W_o, d, attn->z, 1, 0.0f, attn->output, 1);
}

Attention* deserialize_attention(FILE* f, int seq_len, int num_heads) {
    int d_model; bool is_causal, use_rope;
    fread(&d_model, sizeof(int), 1, f);
    fread(&is_causal, sizeof(bool), 1, f);
    fread(&use_rope, sizeof(bool), 1, f);
    
    Attention* attn = init_attention(seq_len, d_model, num_heads, is_causal, use_rope);
    size_t w_size = d_model * d_model;
    
    fread(attn->W_q, sizeof(float), w_size, f);
    fread(attn->W_k, sizeof(float), w_size, f);
    fread(attn->W_v, sizeof(float), w_size, f);
    fread(attn->W_o, sizeof(float), w_size, f);
    
    // Skip optimizer state
    int t; fread(&t, sizeof(int), 1, f);
    fseek(f, 8 * w_size * sizeof(float), SEEK_CUR);
    
    return attn;
}

// ============================================================================
// Transformer Functions
// ============================================================================

Transformer* init_transformer(int seq_len, int d_model, int hidden_dim, int num_layers) {
    Transformer* t = (Transformer*)malloc(sizeof(Transformer));
    t->seq_len = seq_len; t->d_model = d_model;
    t->hidden_dim = hidden_dim; t->num_layers = num_layers;
    t->attention_layers = (Attention**)malloc(num_layers * sizeof(Attention*));
    t->mlp_layers = (MLP**)malloc(num_layers * sizeof(MLP*));
    t->norm1 = (float*)malloc(d_model * sizeof(float));
    t->norm2 = (float*)malloc(d_model * sizeof(float));
    return t;
}

void free_transformer(Transformer* t) {
    for (int i = 0; i < t->num_layers; i++) {
        free_attention(t->attention_layers[i]);
        free_mlp(t->mlp_layers[i]);
    }
    free(t->attention_layers); free(t->mlp_layers);
    free(t->norm1); free(t->norm2);
    free(t);
}

static void rmsnorm(float* out, float* in, int d) {
    float sum_sq = 0.0f;
    for (int i = 0; i < d; i++) sum_sq += in[i] * in[i];
    float scale = 1.0f / sqrtf(sum_sq / d + 1e-6f);
    for (int i = 0; i < d; i++) out[i] = in[i] * scale;
}

void forward_transformer(Transformer* t, float* x, int pos) {
    int d = t->d_model;
    
    for (int layer = 0; layer < t->num_layers; layer++) {
        // Pre-attention RMSNorm
        rmsnorm(t->norm1, x, d);
        
        // Attention with KV cache
        forward_attention(t->attention_layers[layer], t->norm1, pos);
        
        // Residual connection
        for (int i = 0; i < d; i++) x[i] += t->attention_layers[layer]->output[i];
        
        // Pre-MLP RMSNorm
        rmsnorm(t->norm2, x, d);
        
        // MLP
        forward_mlp(t->mlp_layers[layer], t->norm2);
        
        // Residual connection
        for (int i = 0; i < d; i++) x[i] += t->mlp_layers[layer]->output[i];
    }
}

Transformer* deserialize_transformer(FILE* f, int seq_len) {
    int d_model, hidden_dim, num_layers;
    bool is_causal, use_rope;
    fread(&d_model, sizeof(int), 1, f);
    fread(&hidden_dim, sizeof(int), 1, f);
    fread(&num_layers, sizeof(int), 1, f);
    fread(&is_causal, sizeof(bool), 1, f);
    fread(&use_rope, sizeof(bool), 1, f);
    (void)is_causal; (void)use_rope;
    
    Transformer* t = init_transformer(seq_len, d_model, hidden_dim, num_layers);
    
    for (int i = 0; i < num_layers; i++) {
        t->attention_layers[i] = deserialize_attention(f, seq_len, 8);
        t->mlp_layers[i] = deserialize_mlp(f);
    }
    return t;
}

// ============================================================================
// GPT Functions
// ============================================================================

GPT* init_gpt(int seq_len, int d_model, int hidden_dim, int num_layers, int vocab_size) {
    GPT* gpt = (GPT*)malloc(sizeof(GPT));
    gpt->seq_len = seq_len; gpt->d_model = d_model;
    gpt->hidden_dim = hidden_dim; gpt->num_layers = num_layers;
    gpt->vocab_size = vocab_size;
    gpt->token_embedding = (float*)malloc((size_t)vocab_size * d_model * sizeof(float));
    gpt->x = (float*)malloc(d_model * sizeof(float));
    gpt->final_norm = (float*)malloc(d_model * sizeof(float));
    return gpt;
}

void free_gpt(GPT* gpt) {
    if (!gpt) return;
    free(gpt->token_embedding);
    free(gpt->x); free(gpt->final_norm);
    free_transformer(gpt->transformer);
    free(gpt);
}

void forward_gpt(GPT* gpt, unsigned short token, int pos, float* logits) {
    int d = gpt->d_model;
    
    // Token embedding
    memcpy(gpt->x, gpt->token_embedding + token * d, d * sizeof(float));
    
    // Transformer with KV cache
    forward_transformer(gpt->transformer, gpt->x, pos);
    
    // Final RMSNorm
    rmsnorm(gpt->final_norm, gpt->x, d);
    
    // Output projection: logits = token_embedding @ final_norm
    cblas_sgemv(CblasRowMajor, CblasNoTrans, gpt->vocab_size, d,
                1.0f, gpt->token_embedding, d, gpt->final_norm, 1, 0.0f, logits, 1);
}

GPT* load_gpt(const char* filename, int seq_len) {
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("Error opening file: %s\n", filename); return NULL; }
    
    int d_model, hidden_dim, num_layers, vocab_size;
    fread(&d_model, sizeof(int), 1, f);
    fread(&hidden_dim, sizeof(int), 1, f);
    fread(&num_layers, sizeof(int), 1, f);
    fread(&vocab_size, sizeof(int), 1, f);
    
    GPT* gpt = init_gpt(seq_len, d_model, hidden_dim, num_layers, vocab_size);
    
    size_t emb_size = (size_t)vocab_size * d_model;
    fread(gpt->token_embedding, sizeof(float), emb_size, f);
    
    // Skip optimizer state
    int t; fread(&t, sizeof(int), 1, f);
    fseek(f, 2 * emb_size * sizeof(float), SEEK_CUR);
    
    gpt->transformer = deserialize_transformer(f, seq_len);
    
    fclose(f);
    return gpt;
}

// ============================================================================
// Chat Application
// ============================================================================

static GPT* global_gpt = NULL;
static unsigned short* global_tokens = NULL;
static float* global_logits = NULL;

void cleanup_and_exit(int signum) {
    (void)signum;
    printf("\n\nExiting...\n");
    free(global_tokens);
    free(global_logits);
    free_gpt(global_gpt);
    exit(0);
}

unsigned short sample_token(float* logits, int vocab_size, float temperature) {
    float max_logit = -1e30f;
    for (int v = 0; v < vocab_size; v++) {
        logits[v] /= temperature;
        if (logits[v] > max_logit) max_logit = logits[v];
    }
    
    float sum_exp = 0.0f;
    for (int v = 0; v < vocab_size; v++) {
        logits[v] = expf(logits[v] - max_logit);
        sum_exp += logits[v];
    }
    
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int v = 0; v < vocab_size; v++) {
        cumsum += logits[v] / sum_exp;
        if (r <= cumsum) return (unsigned short)v;
    }
    return (unsigned short)(vocab_size - 1);
}

void generate_response(GPT* gpt, const char* question, unsigned short* tokens, float* logits) {
    const int seq_len = gpt->seq_len;
    const float temperature = 0.7f;
    const int max_new_tokens = 256;
    const char* end_marker = "<|assistant_end|>";
    const size_t end_marker_len = strlen(end_marker);
    
    // Build prompt
    char prompt[4096];
    snprintf(prompt, sizeof(prompt), "<|bos|><|user_start|>%s<|user_end|><|assistant_start|>", question);
    size_t prompt_len = strlen(prompt);
    int prompt_token_count = (prompt_len + 1) / 2;
    
    if (prompt_token_count >= seq_len) {
        printf("Prompt too long!\n");
        return;
    }
    
    // Encode prompt tokens
    memset(tokens, 0, seq_len * sizeof(unsigned short));
    for (int i = 0; i < prompt_token_count; i++) {
        unsigned char hi = (unsigned char)prompt[i * 2];
        unsigned char lo = ((size_t)(i * 2 + 1) < prompt_len) ? (unsigned char)prompt[i * 2 + 1] : ' ';
        tokens[i] = (unsigned short)((hi << 8) | lo);
    }
    
    // Prefill: process all prompt tokens
    int pos = prompt_token_count - 1;
    for (int t = 0; t <= pos; t++) {
        forward_gpt(gpt, tokens[t], t, logits);
    }
    
    // Generation loop
    char output_buffer[2048];
    int output_len = 0, printed_len = 0, done = 0;
    
    for (int gen = 0; gen < max_new_tokens && pos < seq_len - 1 && !done; gen++) {
        unsigned short next_token = sample_token(logits, gpt->vocab_size, temperature);
        tokens[pos + 1] = next_token;
        
        // Add to output buffer
        if (output_len < (int)sizeof(output_buffer) - 3) {
            output_buffer[output_len++] = (char)(next_token >> 8);
            output_buffer[output_len++] = (char)(next_token & 0xFF);
            output_buffer[output_len] = '\0';
        }
        
        // Check for end marker
        char* marker_pos = strstr(output_buffer, end_marker);
        if (marker_pos) {
            int pos_in_buffer = marker_pos - output_buffer;
            while (printed_len < pos_in_buffer) putchar(output_buffer[printed_len++]);
            done = 1;
            break;
        }
        
        // Print safe characters
        while (printed_len < output_len - (int)end_marker_len && printed_len < output_len) {
            putchar(output_buffer[printed_len++]);
            fflush(stdout);
        }
        
        // Generate next token's logits
        pos++;
        if (pos < seq_len)
            forward_gpt(gpt, tokens[pos], pos, logits);
    }
    
    if (!done) while (printed_len < output_len) putchar(output_buffer[printed_len++]);
    printf("\n");
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    signal(SIGINT, cleanup_and_exit);
    
    if (argc <= 1) {
        fprintf(stderr, "Usage: %s <model_file.bin>\n", argv[0]);
        return 1;
    }
    
    const int seq_len = 128;
    global_gpt = load_gpt(argv[1], seq_len);
    if (!global_gpt) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    global_tokens = (unsigned short*)calloc(seq_len, sizeof(unsigned short));
    global_logits = (float*)malloc(global_gpt->vocab_size * sizeof(float));
    
    char question[4096];
    while (1) {
        printf("\n\033[1;36m?\033[0m ");
        fflush(stdout);
        if (!fgets(question, sizeof(question), stdin)) break;
        question[strcspn(question, "\n")] = 0;
        if (strlen(question) == 0) continue;
        
        printf("\033[1;32m>\033[0m ");
        fflush(stdout);
        generate_response(global_gpt, question, global_tokens, global_logits);
    }
    
    cleanup_and_exit(0);
    return 0;
}