/* ds4-dump-activations — capture per-layer, per-token activation tensors
 * from ds4's canonical CPU forward path. The output tree serves as the
 * oracle for Phase 2 kernel ports: each ported HIP kernel validates its
 * output against the matching tensor in this dump.
 *
 * Usage: ds4-dump-activations MODEL.gguf "PROMPT" OUT_DIR [N_TOKENS]
 *
 * Backend is forced to CPU — that's the only path with activation hooks
 * (the GPU path doesn't go through layer_forward_raw_swa_one). To capture
 * BOTH prefill and decode positions through the same hooks, we bypass
 * ds4_session_sync (which uses a batched prefill path that skips the
 * single-token forward) and instead call ds4_session_eval for every
 * prompt token AND every generated token.
 *
 * Layout:
 *   OUT_DIR/L{LL}/T{TTTT}/{tag}.bin       activations, raw bytes
 *   OUT_DIR/L{LL}/weight/{tag}.bin        weights (deduped, fired at pos=0)
 *   OUT_DIR/manifest.json                  tensor index + run metadata
 *   OUT_DIR/tokens.json                    prompt + generated tokens
 *   OUT_DIR/logits.f32                     per-token logits (51 rows × vocab f32)
 */

#include "../ds4/ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <stdint.h>
#include <inttypes.h>

static int mkdir_p(const char *path) {
    /* Create directory and all parents. Idempotent. */
    char buf[4096];
    size_t n = strnlen(path, sizeof(buf) - 1);
    if (n == 0) return 0;
    memcpy(buf, path, n);
    buf[n] = '\0';
    for (size_t i = 1; i <= n; i++) {
        if (buf[i] == '/' || buf[i] == '\0') {
            char saved = buf[i];
            buf[i] = '\0';
            if (mkdir(buf, 0755) != 0 && errno != EEXIST) {
                fprintf(stderr, "ds4-dump-activations: mkdir %s: %s\n", buf, strerror(errno));
                return -1;
            }
            buf[i] = saved;
        }
    }
    return 0;
}

/* Bytes per element for each public dtype. */
static size_t dtype_bytes(ds4_dump_dtype dt) {
    switch (dt) {
        case DS4_DUMP_DTYPE_F32: return 4;
        case DS4_DUMP_DTYPE_F16: return 2;
        case DS4_DUMP_DTYPE_FP8: return 1;
        case DS4_DUMP_DTYPE_I32: return 4;
    }
    return 0;
}
static const char *dtype_name(ds4_dump_dtype dt) {
    switch (dt) {
        case DS4_DUMP_DTYPE_F32: return "f32";
        case DS4_DUMP_DTYPE_F16: return "f16";
        case DS4_DUMP_DTYPE_FP8: return "fp8";
        case DS4_DUMP_DTYPE_I32: return "i32";
    }
    return "?";
}

/* Max layer index seen via the dump callback. ds4 has DS4_N_LAYER=43 real
 * layers (indices 0..42), plus one synthetic "layer 43" used by the 0003
 * patch as the head/tail bucket. Sized with headroom for future synthetic
 * buckets if more emerge. */
#define DUMP_MAX_LAYERS 64

struct dump_state {
    const char *out_dir;
    FILE *manifest;          /* the running JSON array (one line per tensor) */
    int manifest_count;       /* number of tensors emitted so far */
    char weight_seen[64][DUMP_MAX_LAYERS]; /* (tag_hash_low_bits, layer) bitmap for weight dedup */
};

/* Tiny hash for weight tag dedup. Tags are stable short strings. */
static unsigned weight_hash(const char *s) {
    unsigned h = 2166136261u;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        h ^= *p;
        h *= 16777619u;
    }
    return h % 64;
}

/* Write s into f with JSON-string escaping. */
static void fprint_json_string(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
            case '"':  fputs("\\\"", f); break;
            case '\\': fputs("\\\\", f); break;
            case '\n': fputs("\\n", f); break;
            case '\r': fputs("\\r", f); break;
            case '\t': fputs("\\t", f); break;
            default:
                if (c < 0x20) fprintf(f, "\\u%04x", c);
                else fputc(c, f);
        }
    }
    fputc('"', f);
}

static void on_activation(void *ud, const char *tag, const void *data,
                          ds4_dump_dtype dtype, int ndim, const int64_t *shape,
                          int layer, int token_pos) {
    struct dump_state *st = (struct dump_state *)ud;

    /* Compute element count + total bytes. */
    int64_t n_elem = 1;
    for (int i = 0; i < ndim; i++) n_elem *= shape[i];
    size_t bytes = (size_t)n_elem * dtype_bytes(dtype);

    /* Distinguish weight tags (deduped) from per-token activations. */
    int is_weight = (strncmp(tag, "weight:", 7) == 0);
    const char *short_tag = is_weight ? tag + 7 : tag;

    if (is_weight) {
        /* Dedup: emit once per (layer, tag). */
        if (layer < 0 || layer >= DUMP_MAX_LAYERS) return;
        unsigned h = weight_hash(short_tag);
        if (st->weight_seen[h][layer]) return;
        st->weight_seen[h][layer] = 1;
    }

    /* Build paths. out_dir can be a deep absolute path; size accordingly. */
    char dirpath[4096];
    char filepath[8192];
    char relpath[256];
    if (is_weight) {
        snprintf(dirpath, sizeof(dirpath), "%s/L%02d/weight", st->out_dir, layer);
        snprintf(relpath, sizeof(relpath), "L%02d/weight/%s.bin", layer, short_tag);
    } else {
        snprintf(dirpath, sizeof(dirpath), "%s/L%02d/T%04d", st->out_dir, layer, token_pos);
        snprintf(relpath, sizeof(relpath), "L%02d/T%04d/%s.bin", layer, token_pos, short_tag);
    }
    snprintf(filepath, sizeof(filepath), "%s/%s", st->out_dir, relpath);

    if (mkdir_p(dirpath) != 0) return;

    FILE *fp = fopen(filepath, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-dump-activations: open %s: %s\n", filepath, strerror(errno));
        return;
    }
    if (bytes > 0 && fwrite(data, 1, bytes, fp) != bytes) {
        fprintf(stderr, "ds4-dump-activations: short write to %s\n", filepath);
    }
    fclose(fp);

    /* Append manifest entry. Format: one JSON object per line for streaming. */
    if (st->manifest_count > 0) fputs(",\n", st->manifest);
    fputs("    {\"tag\":", st->manifest);
    fprint_json_string(st->manifest, short_tag);
    fprintf(st->manifest, ",\"layer\":%d,\"token\":%d,\"dtype\":\"%s\",\"shape\":[",
            layer, is_weight ? -1 : token_pos, dtype_name(dtype));
    for (int i = 0; i < ndim; i++) {
        fprintf(st->manifest, "%s%" PRId64, i ? "," : "", shape[i]);
    }
    fprintf(st->manifest, "],\"bytes\":%zu,\"path\":", bytes);
    fprint_json_string(st->manifest, relpath);
    fprintf(st->manifest, ",\"is_weight\":%s}", is_weight ? "true" : "false");
    st->manifest_count++;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s MODEL.gguf PROMPT OUTPUT_DIR [N_TOKENS]\n"
            "  Writes a tensor tree + manifest.json + tokens.json + logits.f32.\n"
            "  Backend is forced to CPU (canonical reference path).\n",
            argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *prompt_text = argv[2];
    const char *out_dir = argv[3];
    const int n_tokens = argc >= 5 ? atoi(argv[4]) : 50;

    if (mkdir_p(out_dir) != 0) return 1;

    /* Open manifest file with metadata header now; tensors stream in via callback. */
    char manifest_path[1024];
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.json", out_dir);
    FILE *fp_man = fopen(manifest_path, "w");
    if (!fp_man) {
        fprintf(stderr, "ds4-dump-activations: open %s: %s\n", manifest_path, strerror(errno));
        return 1;
    }
    fputs("{\n  \"meta\": {\n    \"model_path\": ", fp_man);
    fprint_json_string(fp_man, model_path);
    fputs(",\n    \"prompt\": ", fp_man);
    fprint_json_string(fp_man, prompt_text);
    fprintf(fp_man, ",\n    \"n_tokens_arg\": %d,\n    \"backend\": \"cpu\"\n  },\n  \"tensors\": [\n",
            n_tokens);

    struct dump_state state = { .out_dir = out_dir, .manifest = fp_man, .manifest_count = 0 };
    memset(state.weight_seen, 0, sizeof(state.weight_seen));

    /* Open engine on CPU backend. */
    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = DS4_BACKEND_CPU;
    opt.n_threads = 0;

    fprintf(stderr, "ds4-dump-activations: opening engine (cpu)\n");
    ds4_engine *engine = NULL;
    int rc = ds4_engine_open(&engine, &opt);
    if (rc != 0 || !engine) {
        fprintf(stderr, "ds4-dump-activations: ds4_engine_open rc=%d\n", rc);
        fclose(fp_man);
        return 1;
    }

    /* Tokenize prompt with the plain BPE path (same as dump_logits). */
    ds4_tokens prompt = {0};
    ds4_tokenize_text(engine, prompt_text, &prompt);
    if (prompt.len <= 0) {
        fprintf(stderr, "ds4-dump-activations: empty tokenization\n");
        ds4_engine_close(engine);
        fclose(fp_man);
        return 1;
    }
    fprintf(stderr, "ds4-dump-activations: prompt = %d tokens\n", prompt.len);

    /* Create session with enough context. */
    const int ctx_size = prompt.len + n_tokens + 32;
    ds4_session *session = NULL;
    rc = ds4_session_create(&session, engine, ctx_size);
    if (rc != 0 || !session) {
        fprintf(stderr, "ds4-dump-activations: ds4_session_create rc=%d\n", rc);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        fclose(fp_man);
        return 1;
    }

    /* Open the logits/tokens output files (so this is a superset of dump_logits). */
    char path[1024];
    snprintf(path, sizeof(path), "%s/logits.f32", out_dir);
    FILE *fp_logits = fopen(path, "wb");
    snprintf(path, sizeof(path), "%s/tokens.json", out_dir);
    FILE *fp_tokens = fopen(path, "w");
    if (!fp_logits || !fp_tokens) {
        fprintf(stderr, "ds4-dump-activations: open tokens/logits outputs failed\n");
        if (fp_logits) fclose(fp_logits);
        if (fp_tokens) fclose(fp_tokens);
        ds4_session_free(session); ds4_tokens_free(&prompt); ds4_engine_close(engine);
        fclose(fp_man);
        return 1;
    }
    fprintf(fp_tokens, "{\n  \"prompt_tokens\": [");
    for (int i = 0; i < prompt.len; i++) {
        fprintf(fp_tokens, "%s%d", i ? "," : "", prompt.v[i]);
    }
    fprintf(fp_tokens, "],\n  \"generated_tokens\": [");

    /* Register the dump callback ONCE, just before we start feeding tokens. */
    ds4_set_activation_dump(on_activation, &state);

    /* Prefill: feed each prompt token via session_eval so it routes through
     * the canonical single-token CPU forward path (and our hooks fire). */
    char err[256] = {0};
    for (int i = 0; i < prompt.len; i++) {
        rc = ds4_session_eval(session, prompt.v[i], err, sizeof(err));
        if (rc != 0) {
            fprintf(stderr, "ds4-dump-activations: prefill eval(token=%d, pos=%d) failed: %s\n",
                    prompt.v[i], i, err);
            goto fail;
        }
    }

    /* Capture per-token logits the same way dump_logits does. */
    const float *logits_ptr = NULL;
    int vocab_size = 0;
    if (!ds4_session_logits_buffer(session, &logits_ptr, &vocab_size)) {
        fprintf(stderr, "ds4-dump-activations: logits_buffer failed\n");
        goto fail;
    }
    if (fwrite(logits_ptr, sizeof(float), (size_t)vocab_size, fp_logits) != (size_t)vocab_size) {
        fprintf(stderr, "ds4-dump-activations: short logit write (prefill)\n");
        goto fail;
    }

    int eos = ds4_token_eos(engine);
    int gen_count = 0;
    int next_token = ds4_session_argmax(session);

    for (int i = 0; i < n_tokens; i++) {
        fprintf(fp_tokens, "%s%d", i ? "," : "", next_token);
        gen_count++;
        if (next_token == eos) {
            fprintf(stderr, "ds4-dump-activations: EOS at gen %d\n", i);
            break;
        }
        rc = ds4_session_eval(session, next_token, err, sizeof(err));
        if (rc != 0) {
            fprintf(stderr, "ds4-dump-activations: decode eval(token=%d) failed: %s\n",
                    next_token, err);
            goto fail;
        }
        if (!ds4_session_logits_buffer(session, &logits_ptr, &vocab_size)) {
            fprintf(stderr, "ds4-dump-activations: logits_buffer failed at gen %d\n", i);
            goto fail;
        }
        if (fwrite(logits_ptr, sizeof(float), (size_t)vocab_size, fp_logits) != (size_t)vocab_size) {
            fprintf(stderr, "ds4-dump-activations: short logit write (gen %d)\n", i);
            goto fail;
        }
        next_token = ds4_session_argmax(session);
    }

    /* Detach callback before close — no more emits expected. */
    ds4_set_activation_dump(NULL, NULL);

    fprintf(fp_tokens, "],\n  \"vocab_size\": %d,\n  \"backend\": \"cpu\",\n  \"n_logit_rows\": %d\n}\n",
            vocab_size, gen_count + 1);
    fclose(fp_tokens);
    fclose(fp_logits);

    /* Close manifest. */
    fputs("\n  ],\n", fp_man);
    fprintf(fp_man,
            "  \"n_tensors\": %d,\n"
            "  \"n_logit_rows\": %d,\n"
            "  \"vocab_size\": %d,\n"
            "  \"prompt_len\": %d\n"
            "}\n",
            state.manifest_count, gen_count + 1, vocab_size, prompt.len);
    fclose(fp_man);

    fprintf(stderr, "ds4-dump-activations: wrote %d tensors, %d logit rows × %d floats\n",
            state.manifest_count, gen_count + 1, vocab_size);

    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return 0;

fail:
    ds4_set_activation_dump(NULL, NULL);
    fclose(fp_logits);
    fclose(fp_tokens);
    fclose(fp_man);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return 1;
}
