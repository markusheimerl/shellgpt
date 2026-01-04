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
    float* W1_grad; float* W2_grad;
    float* W1_m; float* W1_v;
    float* W2_m; float* W2_v;
    float beta1; float beta2; float epsilon;
    int t; float weight_decay;
    float* preact; float* postact; float* output;
    float* grad_output; float* grad_postact;
    int input_dim; int hidden_dim; int output_dim; int batch_size;
} MLP;

typedef struct {
    float* W_q; float* W_k; float* W_v; float* W_o;
    float* W_q_grad; float* W_k_grad; float* W_v_grad; float* W_o_grad;
    float* W_q_m; float* W_q_v; float* W_k_m; float* W_k_v;
    float* W_v_m; float* W_v_v; float* W_o_m; float* W_o_v;
    float beta1; float beta2; float epsilon; int t; float weight_decay;
    float* Q; float* K; float* V;
    float* scores; float* attn_weights; float* attn_output; float* output;
    float* grad_output; float* grad_attn_output; float* grad_weights;
    float* grad_scores; float* grad_Q; float* grad_K; float* grad_V;
    int seq_len; int d_model; int batch_size;
    int num_heads; int head_dim; float scale;
    bool is_causal; bool use_rope;
} Attention;

typedef struct {
    Attention** attention_layers;
    MLP** mlp_layers;
    float** norm_attn_inputs;
    float** norm_mlp_inputs;
    int seq_len; int d_model; int batch_size;
    int hidden_dim; int num_layers;
} Transformer;

typedef struct {
    float* token_embedding; float* token_embedding_grad;
    float* token_embedding_m; float* token_embedding_v;
    float beta1; float beta2; float epsilon; int t; float weight_decay;
    float* embedded_input; float* norm_output; float* output;
    float* grad_output; float* grad_norm_output;
    Transformer* transformer;
    int seq_len; int d_model; int batch_size;
    int hidden_dim; int num_layers; int vocab_size;
} GPT;

// ============================================================================
// MLP Functions
// ============================================================================

MLP* init_mlp(int input_dim, int hidden_dim, int output_dim, int batch_size) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    mlp->input_dim = input_dim; mlp->hidden_dim = hidden_dim;
    mlp->output_dim = output_dim; mlp->batch_size = batch_size;
    mlp->beta1 = 0.9f; mlp->beta2 = 0.999f; mlp->epsilon = 1e-8f;
    mlp->t = 0; mlp->weight_decay = 0.01f;
    
    size_t w1_size = input_dim * hidden_dim;
    size_t w2_size = hidden_dim * output_dim;
    mlp->W1 = (float*)malloc(w1_size * sizeof(float));
    mlp->W2 = (float*)malloc(w2_size * sizeof(float));
    mlp->W1_grad = (float*)malloc(w1_size * sizeof(float));
    mlp->W2_grad = (float*)malloc(w2_size * sizeof(float));
    mlp->W1_m = (float*)calloc(w1_size, sizeof(float));
    mlp->W1_v = (float*)calloc(w1_size, sizeof(float));
    mlp->W2_m = (float*)calloc(w2_size, sizeof(float));
    mlp->W2_v = (float*)calloc(w2_size, sizeof(float));
    mlp->preact = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    mlp->postact = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    mlp->output = (float*)malloc(batch_size * output_dim * sizeof(float));
    mlp->grad_postact = mlp->postact; mlp->grad_output = mlp->output;
    
    float scale_W1 = 1.0f / sqrtf(input_dim);
    float scale_W2 = 1.0f / sqrtf(hidden_dim);
    for (size_t i = 0; i < w1_size; i++)
        mlp->W1[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_W1;
    for (size_t i = 0; i < w2_size; i++)
        mlp->W2[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_W2;
    return mlp;
}

void free_mlp(MLP* mlp) {
    free(mlp->W1); free(mlp->W2); free(mlp->W1_grad); free(mlp->W2_grad);
    free(mlp->W1_m); free(mlp->W1_v); free(mlp->W2_m); free(mlp->W2_v);
    free(mlp->preact); free(mlp->postact); free(mlp->output); free(mlp);
}

void forward_pass_mlp(MLP* mlp, float* X) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                mlp->batch_size, mlp->hidden_dim, mlp->input_dim,
                1.0f, X, mlp->input_dim, mlp->W1, mlp->hidden_dim,
                0.0f, mlp->preact, mlp->hidden_dim);
    for (int i = 0; i < mlp->batch_size * mlp->hidden_dim; i++)
        mlp->postact[i] = mlp->preact[i] / (1.0f + expf(-mlp->preact[i]));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                mlp->batch_size, mlp->output_dim, mlp->hidden_dim,
                1.0f, mlp->postact, mlp->hidden_dim, mlp->W2, mlp->output_dim,
                0.0f, mlp->output, mlp->output_dim);
}

MLP* deserialize_mlp(FILE* file, int batch_size) {
    int input_dim, hidden_dim, output_dim;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size);
    int w1_size = input_dim * hidden_dim;
    int w2_size = hidden_dim * output_dim;
    fread(mlp->W1, sizeof(float), w1_size, file);
    fread(mlp->W2, sizeof(float), w2_size, file);
    fread(&mlp->t, sizeof(int), 1, file);
    fread(mlp->W1_m, sizeof(float), w1_size, file);
    fread(mlp->W1_v, sizeof(float), w1_size, file);
    fread(mlp->W2_m, sizeof(float), w2_size, file);
    fread(mlp->W2_v, sizeof(float), w2_size, file);
    return mlp;
}

// ============================================================================
// Attention Functions
// ============================================================================

Attention* init_attention(int seq_len, int d_model, int num_heads, int batch_size, bool is_causal, bool use_rope) {
    if (num_heads <= 0 || d_model % num_heads != 0) {
        fprintf(stderr, "d_model must be divisible by num_heads\n");
        exit(EXIT_FAILURE);
    }
    Attention* attn = (Attention*)malloc(sizeof(Attention));
    attn->seq_len = seq_len; attn->d_model = d_model; attn->batch_size = batch_size;
    attn->num_heads = num_heads; attn->head_dim = d_model / num_heads;
    attn->scale = 1.0f / sqrtf(attn->head_dim); attn->is_causal = is_causal; attn->use_rope = use_rope;
    attn->beta1 = 0.9f; attn->beta2 = 0.999f; attn->epsilon = 1e-8f;
    attn->t = 0; attn->weight_decay = 0.01f;
    
    size_t weight_size = d_model * d_model;
    size_t seq_batch_size = batch_size * seq_len * d_model;
    size_t attn_matrix_size = num_heads * batch_size * seq_len * seq_len;
    
    attn->W_q = (float*)malloc(weight_size * sizeof(float));
    attn->W_k = (float*)malloc(weight_size * sizeof(float));
    attn->W_v = (float*)malloc(weight_size * sizeof(float));
    attn->W_o = (float*)malloc(weight_size * sizeof(float));
    attn->W_q_grad = (float*)malloc(weight_size * sizeof(float));
    attn->W_k_grad = (float*)malloc(weight_size * sizeof(float));
    attn->W_v_grad = (float*)malloc(weight_size * sizeof(float));
    attn->W_o_grad = (float*)malloc(weight_size * sizeof(float));
    attn->W_q_m = (float*)calloc(weight_size, sizeof(float));
    attn->W_q_v = (float*)calloc(weight_size, sizeof(float));
    attn->W_k_m = (float*)calloc(weight_size, sizeof(float));
    attn->W_k_v = (float*)calloc(weight_size, sizeof(float));
    attn->W_v_m = (float*)calloc(weight_size, sizeof(float));
    attn->W_v_v = (float*)calloc(weight_size, sizeof(float));
    attn->W_o_m = (float*)calloc(weight_size, sizeof(float));
    attn->W_o_v = (float*)calloc(weight_size, sizeof(float));
    attn->Q = (float*)malloc(seq_batch_size * sizeof(float));
    attn->K = (float*)malloc(seq_batch_size * sizeof(float));
    attn->V = (float*)malloc(seq_batch_size * sizeof(float));
    attn->scores = (float*)malloc(attn_matrix_size * sizeof(float));
    attn->attn_weights = (float*)malloc(attn_matrix_size * sizeof(float));
    attn->attn_output = (float*)malloc(seq_batch_size * sizeof(float));
    attn->output = (float*)malloc(seq_batch_size * sizeof(float));
    attn->grad_output = attn->output; attn->grad_attn_output = attn->attn_output;
    attn->grad_weights = (float*)malloc(attn_matrix_size * sizeof(float));
    attn->grad_scores = attn->scores;
    attn->grad_Q = attn->attn_output; attn->grad_K = attn->K; attn->grad_V = attn->V;
    
    float scale_W = 1.0f / sqrtf(d_model);
    for (size_t i = 0; i < weight_size; i++) {
        attn->W_q[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_W;
        attn->W_k[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_W;
        attn->W_v[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_W;
        attn->W_o[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_W;
    }
    return attn;
}

void free_attention(Attention* attn) {
    free(attn->W_q); free(attn->W_k); free(attn->W_v); free(attn->W_o);
    free(attn->W_q_grad); free(attn->W_k_grad); free(attn->W_v_grad); free(attn->W_o_grad);
    free(attn->W_q_m); free(attn->W_q_v); free(attn->W_k_m); free(attn->W_k_v);
    free(attn->W_v_m); free(attn->W_v_v); free(attn->W_o_m); free(attn->W_o_v);
    free(attn->Q); free(attn->K); free(attn->V);
    free(attn->scores); free(attn->attn_weights);
    free(attn->attn_output); free(attn->output);
    free(attn->grad_weights); free(attn);
}

static void softmax_causal_forward(float* weights, float* scores, int num_heads, int batch_size, int seq_len) {
    for (int h = 0; h < num_heads; h++) {
        for (int b = 0; b < batch_size; b++) {
            float* scores_m = &scores[(h * batch_size + b) * seq_len * seq_len];
            float* weights_m = &weights[(h * batch_size + b) * seq_len * seq_len];
            for (int i = 0; i < seq_len; i++) {
                float max_val = -1e30f;
                for (int j = 0; j <= i; j++) {
                    if (scores_m[i * seq_len + j] > max_val) max_val = scores_m[i * seq_len + j];
                }
                float sum_exp = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    if (j <= i) {
                        float exp_val = expf(scores_m[i * seq_len + j] - max_val);
                        weights_m[i * seq_len + j] = exp_val;
                        sum_exp += exp_val;
                    } else weights_m[i * seq_len + j] = 0.0f;
                }
                for (int j = 0; j <= i; j++) weights_m[i * seq_len + j] /= sum_exp;
            }
        }
    }
}

static void rope_forward(float* Q, float* K, int batch_size, int seq_len, int d_model) {
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            for (int d_pair = 0; d_pair < d_model / 2; d_pair++) {
                int d = d_pair * 2;
                float theta = powf(10000.0f, -((float)d / (float)d_model));
                float angle = (float)t * theta;
                float cos_a = cosf(angle), sin_a = sinf(angle);
                int idx = b * seq_len * d_model + t * d_model + d;
                float q0 = Q[idx], q1 = Q[idx + 1];
                Q[idx] = q0 * cos_a - q1 * sin_a;
                Q[idx + 1] = q0 * sin_a + q1 * cos_a;
                float k0 = K[idx], k1 = K[idx + 1];
                K[idx] = k0 * cos_a - k1 * sin_a;
                K[idx + 1] = k0 * sin_a + k1 * cos_a;
            }
        }
    }
}

void forward_pass_attention(Attention* attn, float* X) {
    for (int b = 0; b < attn->batch_size; b++) {
        float* X_b = &X[b * attn->seq_len * attn->d_model];
        float* Q_b = &attn->Q[b * attn->seq_len * attn->d_model];
        float* K_b = &attn->K[b * attn->seq_len * attn->d_model];
        float* V_b = &attn->V[b * attn->seq_len * attn->d_model];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->d_model,
                    1.0f, X_b, attn->d_model, attn->W_q, attn->d_model,
                    0.0f, Q_b, attn->d_model);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->d_model,
                    1.0f, X_b, attn->d_model, attn->W_k, attn->d_model,
                    0.0f, K_b, attn->d_model);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->d_model,
                    1.0f, X_b, attn->d_model, attn->W_v, attn->d_model,
                    0.0f, V_b, attn->d_model);
    }
    if (attn->use_rope) rope_forward(attn->Q, attn->K, attn->batch_size, attn->seq_len, attn->d_model);
    for (int h = 0; h < attn->num_heads; h++) {
        for (int b = 0; b < attn->batch_size; b++) {
            float* Q_hb = &attn->Q[b * attn->seq_len * attn->d_model + h * attn->head_dim];
            float* K_hb = &attn->K[b * attn->seq_len * attn->d_model + h * attn->head_dim];
            float* scores_hb = &attn->scores[(h * attn->batch_size + b) * attn->seq_len * attn->seq_len];
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        attn->seq_len, attn->seq_len, attn->head_dim,
                        attn->scale, Q_hb, attn->d_model, K_hb, attn->d_model,
                        0.0f, scores_hb, attn->seq_len);
        }
    }
    if (attn->is_causal)
        softmax_causal_forward(attn->attn_weights, attn->scores, attn->num_heads, attn->batch_size, attn->seq_len);
    for (int h = 0; h < attn->num_heads; h++) {
        for (int b = 0; b < attn->batch_size; b++) {
            float* weights_hb = &attn->attn_weights[(h * attn->batch_size + b) * attn->seq_len * attn->seq_len];
            float* V_hb = &attn->V[b * attn->seq_len * attn->d_model + h * attn->head_dim];
            float* attn_output_hb = &attn->attn_output[b * attn->seq_len * attn->d_model + h * attn->head_dim];
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        attn->seq_len, attn->head_dim, attn->seq_len,
                        1.0f, weights_hb, attn->seq_len, V_hb, attn->d_model,
                        0.0f, attn_output_hb, attn->d_model);
        }
    }
    for (int b = 0; b < attn->batch_size; b++) {
        float* attn_output_b = &attn->attn_output[b * attn->seq_len * attn->d_model];
        float* output_b = &attn->output[b * attn->seq_len * attn->d_model];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    attn->seq_len, attn->d_model, attn->d_model,
                    1.0f, attn_output_b, attn->d_model, attn->W_o, attn->d_model,
                    0.0f, output_b, attn->d_model);
    }
}

Attention* deserialize_attention(FILE* file, int batch_size, int seq_len, int num_heads) {
    int d_model; bool is_causal, use_rope;
    fread(&d_model, sizeof(int), 1, file);
    fread(&is_causal, sizeof(bool), 1, file);
    fread(&use_rope, sizeof(bool), 1, file);
    Attention* attn = init_attention(seq_len, d_model, num_heads, batch_size, is_causal, use_rope);
    int weight_size = d_model * d_model;
    fread(attn->W_q, sizeof(float), weight_size, file);
    fread(attn->W_k, sizeof(float), weight_size, file);
    fread(attn->W_v, sizeof(float), weight_size, file);
    fread(attn->W_o, sizeof(float), weight_size, file);
    fread(&attn->t, sizeof(int), 1, file);
    fread(attn->W_q_m, sizeof(float), weight_size, file);
    fread(attn->W_q_v, sizeof(float), weight_size, file);
    fread(attn->W_k_m, sizeof(float), weight_size, file);
    fread(attn->W_k_v, sizeof(float), weight_size, file);
    fread(attn->W_v_m, sizeof(float), weight_size, file);
    fread(attn->W_v_v, sizeof(float), weight_size, file);
    fread(attn->W_o_m, sizeof(float), weight_size, file);
    fread(attn->W_o_v, sizeof(float), weight_size, file);
    return attn;
}

// ============================================================================
// Transformer Functions
// ============================================================================

Transformer* init_transformer(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, bool is_causal, bool use_rope) {
    Transformer* t = (Transformer*)malloc(sizeof(Transformer));
    t->seq_len = seq_len; t->d_model = d_model; t->batch_size = batch_size;
    t->hidden_dim = hidden_dim; t->num_layers = num_layers;
    size_t norm_buffer_size = batch_size * seq_len * d_model * sizeof(float);
    t->norm_attn_inputs = (float**)malloc(num_layers * sizeof(float*));
    t->norm_mlp_inputs = (float**)malloc(num_layers * sizeof(float*));
    for (int i = 0; i < num_layers; i++) {
        t->norm_attn_inputs[i] = (float*)malloc(norm_buffer_size);
        t->norm_mlp_inputs[i] = (float*)malloc(norm_buffer_size);
    }
    t->attention_layers = (Attention**)malloc(num_layers * sizeof(Attention*));
    t->mlp_layers = (MLP**)malloc(num_layers * sizeof(MLP*));
    for (int i = 0; i < num_layers; i++) {
        t->attention_layers[i] = init_attention(seq_len, d_model, 8, batch_size, is_causal, use_rope);
        t->mlp_layers[i] = init_mlp(d_model, hidden_dim, d_model, batch_size * seq_len);
    }
    return t;
}

void free_transformer(Transformer* t) {
    for (int i = 0; i < t->num_layers; i++) {
        free_attention(t->attention_layers[i]);
        free_mlp(t->mlp_layers[i]);
    }
    free(t->attention_layers); free(t->mlp_layers);
    for (int i = 0; i < t->num_layers; i++) {
        free(t->norm_attn_inputs[i]); free(t->norm_mlp_inputs[i]);
    }
    free(t->norm_attn_inputs); free(t->norm_mlp_inputs); free(t);
}

static void rmsnorm_forward(float* output, const float* input, int batch_size, int seq_len, int d_model) {
    for (int idx = 0; idx < batch_size * seq_len; idx++) {
        const float* x = &input[idx * d_model];
        float* y = &output[idx * d_model];
        float sum_sq = 0.0f;
        for (int d = 0; d < d_model; d++) sum_sq += x[d] * x[d];
        float rms = sqrtf(sum_sq / d_model + 1e-6f);
        for (int d = 0; d < d_model; d++) y[d] = x[d] / rms;
    }
}

void forward_pass_transformer(Transformer* t, float* X) {
    for (int layer = 0; layer < t->num_layers; layer++) {
        float* layer_input = (layer == 0) ? X : t->mlp_layers[layer-1]->output;
        rmsnorm_forward(t->norm_attn_inputs[layer], layer_input, t->batch_size, t->seq_len, t->d_model);
        forward_pass_attention(t->attention_layers[layer], t->norm_attn_inputs[layer]);
        cblas_saxpy(t->batch_size * t->seq_len * t->d_model, 1.0f, layer_input, 1,
                    t->attention_layers[layer]->output, 1);
        rmsnorm_forward(t->norm_mlp_inputs[layer], t->attention_layers[layer]->output,
                       t->batch_size, t->seq_len, t->d_model);
        forward_pass_mlp(t->mlp_layers[layer], t->norm_mlp_inputs[layer]);
        cblas_saxpy(t->batch_size * t->seq_len * t->d_model, 1.0f,
                    t->attention_layers[layer]->output, 1, t->mlp_layers[layer]->output, 1);
    }
}

Transformer* deserialize_transformer(FILE* file, int batch_size, int seq_len) {
    int d_model, hidden_dim, num_layers; bool is_causal, use_rope;
    fread(&d_model, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&is_causal, sizeof(bool), 1, file);
    fread(&use_rope, sizeof(bool), 1, file);
    Transformer* t = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, is_causal, use_rope);
    for (int i = 0; i < num_layers; i++) {
        free_attention(t->attention_layers[i]);
        free_mlp(t->mlp_layers[i]);
        t->attention_layers[i] = deserialize_attention(file, batch_size, seq_len, 8);
        t->mlp_layers[i] = deserialize_mlp(file, batch_size * seq_len);
    }
    return t;
}

// ============================================================================
// GPT Functions
// ============================================================================

GPT* init_gpt(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size) {
    GPT* gpt = (GPT*)malloc(sizeof(GPT));
    gpt->seq_len = seq_len; gpt->d_model = d_model; gpt->batch_size = batch_size;
    gpt->hidden_dim = hidden_dim; gpt->num_layers = num_layers; gpt->vocab_size = 65536;
    gpt->beta1 = 0.9f; gpt->beta2 = 0.999f; gpt->epsilon = 1e-8f;
    gpt->t = 0; gpt->weight_decay = 0.01f;
    
    size_t token_emb_size = gpt->vocab_size * d_model;
    size_t embedded_size = batch_size * seq_len * d_model;
    size_t output_size = batch_size * seq_len * gpt->vocab_size;
    
    gpt->token_embedding = (float*)malloc(token_emb_size * sizeof(float));
    gpt->token_embedding_grad = (float*)malloc(token_emb_size * sizeof(float));
    gpt->token_embedding_m = (float*)calloc(token_emb_size, sizeof(float));
    gpt->token_embedding_v = (float*)calloc(token_emb_size, sizeof(float));
    gpt->embedded_input = (float*)malloc(embedded_size * sizeof(float));
    gpt->norm_output = (float*)malloc(embedded_size * sizeof(float));
    gpt->output = (float*)malloc(output_size * sizeof(float));
    gpt->grad_output = gpt->output;
    gpt->grad_norm_output = (float*)malloc(embedded_size * sizeof(float));
    
    float token_scale = 1.0f / sqrtf(d_model);
    for (size_t i = 0; i < token_emb_size; i++)
        gpt->token_embedding[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * token_scale;
    
    gpt->transformer = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, true, true);
    return gpt;
}

void free_gpt(GPT* gpt) {
    free_transformer(gpt->transformer);
    free(gpt->token_embedding); free(gpt->token_embedding_grad);
    free(gpt->token_embedding_m); free(gpt->token_embedding_v);
    free(gpt->embedded_input); free(gpt->norm_output);
    free(gpt->output); free(gpt->grad_norm_output); free(gpt);
}

static void token_embedding_lookup(float* embedded, float* token_embedding, unsigned short* tokens,
                                   int batch_size, int seq_len, int d_model) {
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            int token = tokens[b * seq_len + t];
            int emb_offset = b * seq_len * d_model + t * d_model;
            int token_emb_offset = token * d_model;
            for (int d = 0; d < d_model; d++)
                embedded[emb_offset + d] = token_embedding[token_emb_offset + d];
        }
    }
}

void forward_pass_gpt(GPT* gpt, unsigned short* input_tokens) {
    token_embedding_lookup(gpt->embedded_input, gpt->token_embedding, input_tokens,
                          gpt->batch_size, gpt->seq_len, gpt->d_model);
    forward_pass_transformer(gpt->transformer, gpt->embedded_input);
    rmsnorm_forward(gpt->norm_output, gpt->transformer->mlp_layers[gpt->num_layers-1]->output,
                    gpt->batch_size, gpt->seq_len, gpt->d_model);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                gpt->batch_size * gpt->seq_len, gpt->vocab_size, gpt->d_model,
                1.0f, gpt->norm_output, gpt->d_model,
                gpt->token_embedding, gpt->d_model,
                0.0f, gpt->output, gpt->vocab_size);
}

GPT* load_gpt(const char* filename, int batch_size, int seq_len) {
    FILE* file = fopen(filename, "rb");
    if (!file) { printf("Error opening file: %s\n", filename); return NULL; }
    int d_model, hidden_dim, num_layers, vocab_size;
    fread(&d_model, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&vocab_size, sizeof(int), 1, file);
    GPT* gpt = init_gpt(seq_len, d_model, hidden_dim, num_layers, batch_size);
    int token_emb_size = vocab_size * d_model;
    fread(gpt->token_embedding, sizeof(float), token_emb_size, file);
    fread(&gpt->t, sizeof(int), 1, file);
    fread(gpt->token_embedding_m, sizeof(float), token_emb_size, file);
    fread(gpt->token_embedding_v, sizeof(float), token_emb_size, file);
    free_transformer(gpt->transformer);
    gpt->transformer = deserialize_transformer(file, batch_size, seq_len);
    fclose(file);
    printf("Model loaded from %s\n", filename);
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
    if (global_tokens) free(global_tokens);
    if (global_logits) free(global_logits);
    if (global_gpt) free_gpt(global_gpt);
    exit(0);
}

void generate_response(GPT* gpt, const char* question, unsigned short* tokens, float* logits) {
    const int seq_len = gpt->seq_len;
    char prompt[4096];
    snprintf(prompt, sizeof(prompt), "<|bos|><|user_start|>%s<|user_end|><|assistant_start|>", question);
    size_t prompt_len = strlen(prompt);
    int prompt_token_count = (prompt_len + 1) / 2;
    memset(tokens, 0, seq_len * sizeof(unsigned short));
    for (int i = 0; i < prompt_token_count; i++)
        tokens[i] = (unsigned short)((unsigned char)prompt[i * 2] << 8) |
                    ((size_t)(i * 2 + 1) < prompt_len ? (unsigned char)prompt[i * 2 + 1] : ' ');
    
    const float temperature = 0.7f;
    const int max_new_tokens = 256;
    const char* end_marker = "<|assistant_end|>";
    const size_t end_marker_len = strlen(end_marker);
    char output_buffer[2048];
    int output_len = 0, printed_len = 0, pos_start = prompt_token_count - 1, done = 0;
    
    for (int pos = pos_start; pos < pos_start + max_new_tokens && pos < seq_len - 1 && !done; pos++) {
        forward_pass_gpt(gpt, tokens);
        memcpy(logits, &gpt->output[pos * gpt->vocab_size], gpt->vocab_size * sizeof(float));
        float max_logit = -1e30f;
        for (int v = 0; v < gpt->vocab_size; v++) {
            logits[v] /= temperature;
            if (logits[v] > max_logit) max_logit = logits[v];
        }
        float sum_exp = 0.0f;
        for (int v = 0; v < gpt->vocab_size; v++) {
            logits[v] = expf(logits[v] - max_logit);
            sum_exp += logits[v];
        }
        float r = (float)rand() / RAND_MAX, cumsum = 0.0f;
        unsigned short next_token = 0;
        for (int v = 0; v < gpt->vocab_size; v++) {
            cumsum += logits[v] / sum_exp;
            if (r <= cumsum) { next_token = v; break; }
        }
        tokens[pos + 1] = next_token;
        if (output_len < (int)sizeof(output_buffer) - 3) {
            output_buffer[output_len++] = (char)(next_token >> 8);
            output_buffer[output_len++] = (char)(next_token & 0xFF);
            output_buffer[output_len] = '\0';
        }
        char* marker_pos = strstr(output_buffer, end_marker);
        if (marker_pos) {
            int pos_in_buffer = marker_pos - output_buffer;
            while (printed_len < pos_in_buffer) putchar(output_buffer[printed_len++]);
            done = 1; break;
        }
        while (printed_len < output_len - (int)end_marker_len && printed_len < output_len) {
            putchar(output_buffer[printed_len++]);
            fflush(stdout);
        }
    }
    if (!done) while (printed_len < output_len) putchar(output_buffer[printed_len++]);
    printf("\n");
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    signal(SIGINT, cleanup_and_exit);
    openblas_set_num_threads(8);
    
    if (argc <= 1) {
        fprintf(stderr, "Usage: %s <model_file.bin>\n", argv[0]);
        return 1;
    }
    
    const int seq_len = 128;
    printf("Loading model from %s...\n", argv[1]);
    global_gpt = load_gpt(argv[1], 1, seq_len);
    if (!global_gpt) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("Model loaded. Ready!\n");
    
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