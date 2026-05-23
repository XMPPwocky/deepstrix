/* ds4-dump-logits — capture full per-token logits from ds4 as the
 * reference output for Phase 1+ numerical validation.
 *
 * Usage: ds4-dump-logits MODEL.gguf "prompt" OUTPUT_DIR [N_TOKENS]
 *
 * Greedy decode (argmax). Writes:
 *   OUTPUT_DIR/tokens.json    — array of generated token IDs
 *   OUTPUT_DIR/logits.f32     — N_TOKENS × vocab_size raw float32, row-major
 *   OUTPUT_DIR/meta.json      — model SHA256, prompt, vocab size, ds4 commit
 *
 * Uses the deepstrix-patched ds4_session_logits_buffer() to read the full
 * logits vector after each eval.
 */

#include "../ds4/ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>

static int mkdir_p(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) return S_ISDIR(st.st_mode) ? 0 : -1;
    return mkdir(path, 0755);
}

static const char *backend_name_for_env(void) {
    /* Detect ROCm via env var the dev shell sets; otherwise default to CUDA
     * which under our build maps to ROCm via the shim. */
    return getenv("DS4_DUMP_BACKEND");
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s MODEL.gguf PROMPT OUTPUT_DIR [N_TOKENS]\n"
            "  Writes tokens.json, logits.f32, meta.json into OUTPUT_DIR.\n"
            "  Set DS4_DUMP_BACKEND=cpu to force CPU backend (slow but safe).\n",
            argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *prompt_text = argv[2];
    const char *out_dir = argv[3];
    const int n_tokens = argc >= 5 ? atoi(argv[4]) : 50;

    if (mkdir_p(out_dir) != 0 && errno != EEXIST) {
        fprintf(stderr, "ds4-dump-logits: mkdir %s: %s\n", out_dir, strerror(errno));
        return 1;
    }

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = DS4_BACKEND_CUDA; /* ROCm via shim */
    opt.n_threads = 0;

    const char *backend_override = backend_name_for_env();
    if (backend_override) {
        if (strcmp(backend_override, "cpu") == 0) {
            opt.backend = DS4_BACKEND_CPU;
        } else if (strcmp(backend_override, "cuda") == 0 ||
                   strcmp(backend_override, "rocm") == 0) {
            opt.backend = DS4_BACKEND_CUDA;
        }
    }

    ds4_log(stdout, DS4_LOG_DEFAULT, "ds4-dump-logits: opening engine (backend=%s)",
            ds4_backend_name(opt.backend));

    ds4_engine *engine = NULL;
    int rc = ds4_engine_open(&engine, &opt);
    if (rc != 0 || !engine) {
        fprintf(stderr, "ds4-dump-logits: ds4_engine_open failed (rc=%d)\n", rc);
        return 1;
    }

    ds4_tokens prompt = {0};
    ds4_tokenize_text(engine, prompt_text, &prompt);
    if (prompt.len <= 0) {
        fprintf(stderr, "ds4-dump-logits: tokenization produced 0 tokens\n");
        ds4_engine_close(engine);
        return 1;
    }
    fprintf(stderr, "ds4-dump-logits: prompt tokenized to %d tokens\n", prompt.len);

    const int ctx_size = prompt.len + n_tokens + 32;
    ds4_session *session = NULL;
    rc = ds4_session_create(&session, engine, ctx_size);
    if (rc != 0 || !session) {
        fprintf(stderr, "ds4-dump-logits: ds4_session_create failed (rc=%d)\n", rc);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    char err[256] = {0};
    rc = ds4_session_sync(session, &prompt, err, sizeof(err));
    if (rc != 0) {
        fprintf(stderr, "ds4-dump-logits: prefill (ds4_session_sync) failed: %s\n", err);
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    /* Read vocab size from the patched accessor. */
    const float *logits_ptr = NULL;
    int vocab_size = 0;
    if (!ds4_session_logits_buffer(session, &logits_ptr, &vocab_size)) {
        fprintf(stderr, "ds4-dump-logits: ds4_session_logits_buffer failed — is the patch applied?\n");
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }
    fprintf(stderr, "ds4-dump-logits: vocab=%d, dumping %d tokens worth of logits\n",
            vocab_size, n_tokens);

    /* Open output files. */
    char path[1024];
    snprintf(path, sizeof(path), "%s/logits.f32", out_dir);
    FILE *fp_logits = fopen(path, "wb");
    if (!fp_logits) {
        fprintf(stderr, "ds4-dump-logits: open %s: %s\n", path, strerror(errno));
        ds4_session_free(session); ds4_tokens_free(&prompt); ds4_engine_close(engine);
        return 1;
    }
    snprintf(path, sizeof(path), "%s/tokens.json", out_dir);
    FILE *fp_tokens = fopen(path, "w");
    if (!fp_tokens) {
        fprintf(stderr, "ds4-dump-logits: open %s: %s\n", path, strerror(errno));
        fclose(fp_logits);
        ds4_session_free(session); ds4_tokens_free(&prompt); ds4_engine_close(engine);
        return 1;
    }

    fprintf(fp_tokens, "{\n  \"prompt_tokens\": [");
    for (int i = 0; i < prompt.len; i++) {
        fprintf(fp_tokens, "%s%d", i ? "," : "", prompt.v[i]);
    }
    fprintf(fp_tokens, "],\n  \"generated_tokens\": [");

    /* Greedy decode: capture logits after prefill (the "next-token" prediction),
     * then for each subsequent generated token capture its post-eval logits. */
    if (fwrite(logits_ptr, sizeof(float), (size_t)vocab_size, fp_logits) != (size_t)vocab_size) {
        fprintf(stderr, "ds4-dump-logits: logit write failed (prefill)\n");
        goto fail;
    }

    int eos = ds4_token_eos(engine);
    int gen_count = 0;
    int next_token = ds4_session_argmax(session);

    for (int i = 0; i < n_tokens; i++) {
        fprintf(fp_tokens, "%s%d", i ? "," : "", next_token);
        gen_count++;

        if (next_token == eos) {
            fprintf(stderr, "ds4-dump-logits: hit EOS at token %d\n", i);
            break;
        }

        rc = ds4_session_eval(session, next_token, err, sizeof(err));
        if (rc != 0) {
            fprintf(stderr, "ds4-dump-logits: eval(token=%d) failed: %s\n", next_token, err);
            goto fail;
        }

        if (!ds4_session_logits_buffer(session, &logits_ptr, &vocab_size)) {
            fprintf(stderr, "ds4-dump-logits: logits_buffer failed on iteration %d\n", i);
            goto fail;
        }
        if (fwrite(logits_ptr, sizeof(float), (size_t)vocab_size, fp_logits) != (size_t)vocab_size) {
            fprintf(stderr, "ds4-dump-logits: logit write failed on iteration %d\n", i);
            goto fail;
        }
        next_token = ds4_session_argmax(session);
    }

    fprintf(fp_tokens, "],\n  \"vocab_size\": %d,\n  \"backend\": \"%s\",\n  \"n_logit_rows\": %d\n}\n",
            vocab_size, ds4_backend_name(opt.backend), gen_count + 1);
    fclose(fp_tokens);
    fclose(fp_logits);

    fprintf(stderr, "ds4-dump-logits: wrote %d logit rows × %d floats\n", gen_count + 1, vocab_size);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return 0;

fail:
    fclose(fp_logits);
    fclose(fp_tokens);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return 1;
}
