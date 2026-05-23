/* ds4-dump-tokenize — capture ds4's plain-BPE tokenization of a text
 * input so we can byte-compare against our Rust v4flash-core BpeVocab.
 *
 * Usage: ds4-dump-tokenize MODEL.gguf "text" OUTPUT_DIR
 *
 * Writes OUTPUT_DIR/tokens.json:
 *   {"prompt": "...", "token_ids": [int, int, ...]}
 *
 * Implementation: calls ds4_dump_text_tokenization() which loads only the
 * vocab (no weights — fast) and runs tokenize_rendered_chat_vocab. For text
 * containing no special markers (<｜...｜> / <think>), that path is the
 * identity wrapper around bpe_tokenize_text — i.e. plain BPE. We capture
 * its first output line (a JSON array of token IDs) and re-emit it inside
 * a {prompt, token_ids} object.
 *
 * Pre-condition for clean comparison: input text must contain no ds4
 * special markers. Use plain ASCII/UTF-8 prose to validate the BPE path.
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

/* Write s into f with JSON-string escaping. */
static void fprint_json_string(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
            case '"':  fputs("\\\"", f); break;
            case '\\': fputs("\\\\", f); break;
            case '\b': fputs("\\b", f); break;
            case '\f': fputs("\\f", f); break;
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

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s MODEL.gguf TEXT OUTPUT_DIR\n"
            "  Writes OUTPUT_DIR/tokens.json with ds4's BPE tokenization.\n"
            "  TEXT should contain no special markers for a clean BPE comparison.\n",
            argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *text = argv[2];
    const char *out_dir = argv[3];

    if (mkdir_p(out_dir) != 0 && errno != EEXIST) {
        fprintf(stderr, "ds4-dump-tokenize: mkdir %s: %s\n", out_dir, strerror(errno));
        return 1;
    }

    /* Capture ds4_dump_text_tokenization's output into a memstream so we
     * can pull out the first line (the token-id JSON array). */
    char *buf = NULL;
    size_t buflen = 0;
    FILE *mem = open_memstream(&buf, &buflen);
    if (!mem) {
        fprintf(stderr, "ds4-dump-tokenize: open_memstream: %s\n", strerror(errno));
        return 1;
    }
    int rc = ds4_dump_text_tokenization(model_path, text, mem);
    fflush(mem);
    fclose(mem);
    if (rc != 0 || !buf) {
        fprintf(stderr, "ds4-dump-tokenize: ds4_dump_text_tokenization rc=%d\n", rc);
        free(buf);
        return 1;
    }

    /* First line of buf is "[id, id, ...]\n" — copy verbatim. */
    char *nl = strchr(buf, '\n');
    if (!nl) {
        fprintf(stderr, "ds4-dump-tokenize: no newline in output; got %zu bytes\n", buflen);
        free(buf);
        return 1;
    }
    *nl = '\0';

    char path[1024];
    snprintf(path, sizeof(path), "%s/tokens.json", out_dir);
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "ds4-dump-tokenize: open %s: %s\n", path, strerror(errno));
        free(buf);
        return 1;
    }

    fputs("{\n  \"prompt\": ", fp);
    fprint_json_string(fp, text);
    fputs(",\n  \"token_ids\": ", fp);
    fputs(buf, fp);
    fputs("\n}\n", fp);
    fclose(fp);

    fprintf(stderr, "ds4-dump-tokenize: wrote %s (text=%zu chars, %s)\n",
            path, strlen(text), buf);
    free(buf);
    return 0;
}
