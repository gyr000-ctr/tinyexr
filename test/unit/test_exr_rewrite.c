/*
 * TinyEXR - header-only rewrite (exr_rewrite_to_sink/file) regressions.
 *
 * The rewrite path re-serializes headers and copies every compressed chunk
 * byte-for-byte, rebuilding offset tables. These tests pin:
 *   - chunk-level byte identity for unchanged parts (scanline/tiled/deep,
 *     single and multipart, mipmap/ripmap),
 *   - attribute edits land in the new header without touching pixel bytes,
 *   - layout-changing edits get EXR_ERROR_LAYOUT_CHANGED,
 *   - non-seekable sinks get EXR_ERROR_NOT_SEEKABLE,
 *   - the output reloads through the normal reader with identical pixels.
 *
 * Copyright (c) 2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "exr_internal.h"
#include <stdio.h>
#include <stdlib.h>

static int failed, passed;
#define REQUIRE(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++failed; return; } } while (0)
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++failed; } else ++passed; } while (0)

/* ---- whole-file loader (small fixtures only) ----------------------------- */

static uint8_t *read_file(const char *path, size_t *out_size) {
    FILE *fp = fopen(path, "rb");
    long n;
    uint8_t *buf;
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    n = ftell(fp);
    if (n < 0) { fclose(fp); return NULL; }
    rewind(fp);
    buf = (uint8_t *)malloc((size_t)n ? (size_t)n : 1);
    if (!buf) { fclose(fp); return NULL; }
    if (n && fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        free(buf); fclose(fp); return NULL;
    }
    fclose(fp);
    *out_size = (size_t)n;
    return buf;
}

/* ---- seekable growable-memory sink --------------------------------------- */

typedef struct {
    uint8_t *data;
    size_t size, cap, pos;
    int closes;
} mem_sink;

static exr_result mem_sink_write(void *u, const void *p, size_t n) {
    mem_sink *s = (mem_sink *)u;
    if (s->pos > SIZE_MAX - n) return EXR_ERROR_LIMIT_EXCEEDED;
    if (s->pos + n > s->cap) {
        size_t ncap = s->cap ? s->cap : 256;
        uint8_t *nb;
        while (ncap < s->pos + n) ncap *= 2;
        nb = (uint8_t *)realloc(s->data, ncap);
        if (!nb) return EXR_ERROR_OUT_OF_MEMORY;
        s->data = nb; s->cap = ncap;
    }
    if (n) memcpy(s->data + s->pos, p, n);
    s->pos += n;
    if (s->pos > s->size) s->size = s->pos;
    return EXR_SUCCESS;
}
static exr_result mem_sink_seek(void *u, uint64_t off) {
    mem_sink *s = (mem_sink *)u;
    if (off > s->cap) return EXR_ERROR_IO;
    s->pos = (size_t)off;
    return EXR_SUCCESS;
}
static exr_result mem_sink_close(void *u) { ++((mem_sink *)u)->closes; return EXR_SUCCESS; }

static void mem_sink_free(mem_sink *s) { free(s->data); memset(s, 0, sizeof(*s)); }

/* A write-only sink: seek is absent, exercising EXR_ERROR_NOT_SEEKABLE. */
typedef struct { size_t total; int closes; } no_seek_sink;
static exr_result no_seek_write(void *u, const void *p, size_t n) {
    (void)p; ((no_seek_sink *)u)->total += n; return EXR_SUCCESS;
}
static exr_result no_seek_close(void *u) { ++((no_seek_sink *)u)->closes; return EXR_SUCCESS; }

/* ---- random-access data source over an immutable buffer ------------------ */

typedef struct {
    const uint8_t *bytes;
    uint64_t size;
    size_t requests;
    size_t max_request;
} buf_source;

static exr_result buf_source_read(void *u, uint64_t off, uint64_t len,
                                  void *dst) {
    buf_source *s = (buf_source *)u;
    if (off > s->size || len > s->size - off) return EXR_ERROR_CORRUPT;
    if (len > s->max_request) s->max_request = (size_t)len;
    ++s->requests;
    memcpy(dst, s->bytes + (size_t)off, (size_t)len);
    return EXR_SUCCESS;
}

/* A source whose first read always suspends (WOULD_BLOCK). Rewrite has no
 * supply/resume loop, so it must reject this distinctly rather than emit a
 * torn file or leak the success-class WOULD_BLOCK code. */
static exr_result blocking_source_read(void *u, uint64_t off, uint64_t len,
                                       void *dst) {
    (void)u; (void)off; (void)len; (void)dst;
    return EXR_WOULD_BLOCK;
}

/* ---- helpers -------------------------------------------------------------- */

/* Open + parse a reader on an in-memory fixture. */
static exr_result open_memory_reader(const uint8_t *data, size_t size,
                                     exr_reader **out) {
    exr_result rc = exr_reader_open_memory(data, size, NULL, out);
    if (rc != EXR_SUCCESS) return rc;
    return exr_reader_parse_header(*out);
}

/* Chunk on-disk size measured exactly as the rewrite core does it. Mirrors
 * rewrite_measure_chunk so the test can locate each chunk in the output. */
static exr_result chunk_size(const exr_reader *r, int32_t part, uint32_t idx,
                             size_t *out) {
    const exr_int_part *p = &r->parts[part];
    const exr_header *h = &p->header;
    int deep = h->part_type == EXR_PART_DEEP_SCANLINE ||
               h->part_type == EXR_PART_DEEP_TILED;
    size_t hsz = deep ? (h->tiled ? 40u : 28u) : (h->tiled ? 20u : 8u);
    uint64_t n, samples = 0;
    const uint8_t *hdr;
    exr_result rc;
    if (r->is_multipart) hsz += 4;
    rc = exr_reader_fetch((exr_reader *)r, p->offsets[idx], hsz, NULL, &hdr);
    if (rc != EXR_SUCCESS) return rc;
    if (r->is_multipart) hdr += 4;
    if (deep) {
        n = exr_rd_u64(hdr + (h->tiled ? 16 : 4));
        samples = exr_rd_u64(hdr + (h->tiled ? 24 : 12));
    } else {
        int32_t sz = exr_rd_i32(hdr + (h->tiled ? 16 : 4));
        if (sz < 0) return EXR_ERROR_CORRUPT;
        n = (uint32_t)sz;
    }
    if (n > UINT64_MAX - hsz || samples > UINT64_MAX - n - hsz)
        return EXR_ERROR_CORRUPT;
    *out = (size_t)(n + hsz + samples);
    return EXR_SUCCESS;
}

/* Run a rewrite with NULL headers (pure passthrough) and return the produced
 * blob. Also asserts every chunk in the output is byte-identical to the
 * corresponding source chunk. */
static void passthrough_identity(const char *path, int use_source) {
    uint8_t *data;
    size_t size;
    exr_reader *r = NULL;
    mem_sink sink;
    exr_result rc;
    int32_t p;
    buf_source bs;
    exr_data_source ds;

    memset(&sink, 0, sizeof(sink));
    data = read_file(path, &size);
    REQUIRE(data != NULL);

    if (use_source) {
        memset(&bs, 0, sizeof(bs));
        bs.bytes = data; bs.size = size;
        ds.user = &bs; ds.read = buf_source_read; ds.total_size = size;
        rc = exr_reader_open_source(&ds, NULL, &r);
        REQUIRE(rc == EXR_SUCCESS);
        rc = exr_reader_parse_header(r);
    } else {
        rc = open_memory_reader(data, size, &r);
    }
    REQUIRE(rc == EXR_SUCCESS);

    {
        exr_data_sink sk = {&sink, mem_sink_write, mem_sink_seek, mem_sink_close};
        rc = exr_rewrite_to_sink(r, NULL, &sk);
    }
    CHECK(rc == EXR_SUCCESS);
    /* The core takes the sink once preflight passes and invokes close exactly
     * once (parity with exr_writer_begin_stream/end_stream ownership). */
    CHECK(sink.closes == 1);

    if (rc == EXR_SUCCESS) {
        /* Reopen the output and compare every chunk byte-for-byte. */
        exr_reader *out = NULL;
        REQUIRE(open_memory_reader(sink.data, sink.size, &out) == EXR_SUCCESS);
        CHECK(out->num_parts == r->num_parts);
        for (p = 0; p < r->num_parts; ++p) {
            uint32_t k;
            CHECK(out->parts[p].num_chunks == r->parts[p].num_chunks);
            for (k = 0; k < r->parts[p].num_chunks; ++k) {
                size_t n;
                const uint8_t *a, *b;
                REQUIRE(chunk_size(r, p, k, &n) == EXR_SUCCESS);
                REQUIRE(chunk_size(out, p, k, &n) == EXR_SUCCESS);
                REQUIRE(exr_reader_fetch(r, r->parts[p].offsets[k], n, NULL,
                                         &a) == EXR_SUCCESS);
                REQUIRE(exr_reader_fetch(out, out->parts[p].offsets[k], n,
                                         NULL, &b) == EXR_SUCCESS);
                CHECK(memcmp(a, b, n) == 0);
            }
        }
        exr_reader_close(out);
    }
    exr_reader_close(r);
    mem_sink_free(&sink);
    free(data);
}

/* Decode whole images through the public loader and compare pixel bytes. */
static void pixels_equal(const char *path_a, const char *path_b) {
    exr_image a = {0}, b = {0};
    int32_t p;
    REQUIRE(exr_load_from_file(path_a, NULL, &a) == EXR_SUCCESS);
    REQUIRE(exr_load_from_file(path_b, NULL, &b) == EXR_SUCCESS);
    CHECK(a.num_parts == b.num_parts);
    for (p = 0; p < a.num_parts; ++p) {
        int c;
        CHECK(a.parts[p].width == b.parts[p].width);
        CHECK(a.parts[p].height == b.parts[p].height);
        CHECK(a.parts[p].header.num_channels ==
              b.parts[p].header.num_channels);
        for (c = 0; c < a.parts[p].header.num_channels; ++c) {
            const exr_channel *ch = &a.parts[p].header.channels[c];
            size_t bytes = (size_t)a.parts[p].width *
                           (size_t)a.parts[p].height * exr_pixel_size(ch->pixel_type);
            CHECK(a.parts[p].images[c] != NULL && b.parts[p].images[c] != NULL);
            if (a.parts[p].images[c] && b.parts[p].images[c])
                CHECK(memcmp(a.parts[p].images[c], b.parts[p].images[c],
                             bytes) == 0);
        }
    }
    exr_image_free(&a);
    exr_image_free(&b);
}

/* ---- 1. unchanged passthrough across every layout ------------------------- */

static void test_passthrough_all_layouts(void) {
    static const char *fixtures[] = {
        "test/unit/regression/leangen-scanline.exr",
        "test/unit/regression/leangen-tiled.exr",
        "test/unit/regression/leangen-mipmap.exr",
        "test/unit/regression/leangen-ripmap.exr",
        "test/unit/regression/leangen-multipart.exr",
        "test/unit/regression/leangen-deepscanline.exr",
        "test/unit/regression/leangen-deeptiled.exr",
        "test/unit/regression/leangen-deeptiled-edge.exr",
        "test/unit/regression/2by2.exr",
        "test/unit/regression/flaga.exr",
    };
    size_t i;
    for (i = 0; i < sizeof(fixtures) / sizeof(fixtures[0]); ++i) {
        passthrough_identity(fixtures[i], 0);
        passthrough_identity(fixtures[i], 1); /* callback random-access source */
    }
}

/* ---- 2. attribute edit lands, chunks stay identical ----------------------- */

static void test_attribute_edit(void) {
    const char *path = "test/unit/regression/leangen-scanline.exr";
    uint8_t *data;
    size_t size;
    exr_reader *r = NULL;
    mem_sink sink;
    exr_result rc;
    exr_header *edited = NULL;
    const exr_header **hdr_ptrs = NULL;
    char val[64];

    memset(&sink, 0, sizeof(sink));
    data = read_file(path, &size);
    REQUIRE(data != NULL);
    REQUIRE(open_memory_reader(data, size, &r) == EXR_SUCCESS);

    edited = (exr_header *)calloc(1, sizeof(exr_header));
    hdr_ptrs = (const exr_header **)calloc(1, sizeof(exr_header *));
    REQUIRE(edited && hdr_ptrs);
    REQUIRE(exr_header_copy(exr_default_allocator(), edited, exr_reader_part_header(r, 0)) ==
            EXR_SUCCESS);
    hdr_ptrs[0] = edited;

    REQUIRE(exr_header_set_string_attribute(NULL, edited, "rewriteCheck",
                                             "header-only v3") == EXR_SUCCESS);
    REQUIRE(exr_header_get_string_attribute(edited, "rewriteCheck", val,
                                            sizeof(val), NULL) == EXR_SUCCESS);
    CHECK(strcmp(val, "header-only v3") == 0);

    {
        exr_data_sink sk = {&sink, mem_sink_write, mem_sink_seek, mem_sink_close};
        rc = exr_rewrite_to_sink(r, hdr_ptrs, &sk);
    }
    CHECK(rc == EXR_SUCCESS);
    if (rc == EXR_SUCCESS) {
        exr_reader *out = NULL;
        const exr_header *oh;
        REQUIRE(open_memory_reader(sink.data, sink.size, &out) == EXR_SUCCESS);
        oh = exr_reader_part_header(out, 0);
        REQUIRE(oh != NULL);
        CHECK(exr_header_get_string_attribute(oh, "rewriteCheck", val,
                                               sizeof(val), NULL) ==
              EXR_SUCCESS);
        CHECK(strcmp(val, "header-only v3") == 0);
        /* Chunks still byte-identical. */
        {
            uint32_t k;
            for (k = 0; k < r->parts[0].num_chunks; ++k) {
                size_t n;
                const uint8_t *a, *b;
                REQUIRE(chunk_size(r, 0, k, &n) == EXR_SUCCESS);
                REQUIRE(chunk_size(out, 0, k, &n) == EXR_SUCCESS);
                REQUIRE(exr_reader_fetch(r, r->parts[0].offsets[k], n, NULL,
                                         &a) == EXR_SUCCESS);
                REQUIRE(exr_reader_fetch(out, out->parts[0].offsets[k], n,
                                         NULL, &b) == EXR_SUCCESS);
                CHECK(memcmp(a, b, n) == 0);
            }
        }
        exr_reader_close(out);
    }

    exr_header_free(exr_default_allocator(), edited);
    free(edited);
    free(hdr_ptrs);
    exr_reader_close(r);
    mem_sink_free(&sink);
    free(data);
}

/* ---- 3. multipart: edit one part's name, both parts' chunks untouched ----- */

static void test_multipart_edit(void) {
    const char *path = "test/unit/regression/leangen-multipart.exr";
    uint8_t *data;
    size_t size;
    exr_reader *r = NULL;
    mem_sink sink;
    exr_result rc;
    exr_header *edited[4];
    const exr_header *ptrs[4];
    int32_t np, i;

    memset(&sink, 0, sizeof(sink));
    memset(edited, 0, sizeof(edited));
    data = read_file(path, &size);
    REQUIRE(data != NULL);
    REQUIRE(open_memory_reader(data, size, &r) == EXR_SUCCESS);
    np = exr_reader_num_parts(r);
    REQUIRE(np >= 2 && np <= 4);
    for (i = 0; i < np; ++i) {
        edited[i] = (exr_header *)calloc(1, sizeof(exr_header));
        REQUIRE(edited[i]);
        REQUIRE(exr_header_copy(exr_default_allocator(), edited[i],
                                exr_reader_part_header(r, i)) == EXR_SUCCESS);
        ptrs[i] = edited[i];
    }
    /* Rename part 0 (layout-neutral) plus attach a custom attribute. The
     * chunk headers carry the part index, never the name, so chunks survive. */
    snprintf(edited[0]->name, EXR_MAX_NAME, "renamed-part0");
    REQUIRE(exr_header_set_string_attribute(NULL, edited[0], "owner",
                                             "rewrite-suite") == EXR_SUCCESS);

    {
        exr_data_sink sk = {&sink, mem_sink_write, mem_sink_seek, mem_sink_close};
        rc = exr_rewrite_to_sink(r, ptrs, &sk);
    }
    CHECK(rc == EXR_SUCCESS);
    if (rc == EXR_SUCCESS) {
        exr_reader *out = NULL;
        REQUIRE(open_memory_reader(sink.data, sink.size, &out) == EXR_SUCCESS);
        CHECK(exr_reader_num_parts(out) == np);
        CHECK(strcmp(exr_reader_part_header(out, 0)->name, "renamed-part0") ==
              0);
        for (i = 0; i < np; ++i) {
            uint32_t k;
            for (k = 0; k < r->parts[i].num_chunks; ++k) {
                size_t n;
                const uint8_t *a, *b;
                REQUIRE(chunk_size(r, i, k, &n) == EXR_SUCCESS);
                REQUIRE(chunk_size(out, i, k, &n) == EXR_SUCCESS);
                REQUIRE(exr_reader_fetch(r, r->parts[i].offsets[k], n, NULL,
                                         &a) == EXR_SUCCESS);
                REQUIRE(exr_reader_fetch(out, out->parts[i].offsets[k], n,
                                         NULL, &b) == EXR_SUCCESS);
                CHECK(memcmp(a, b, n) == 0);
            }
        }
        exr_reader_close(out);
    }
    for (i = 0; i < np; ++i) { exr_header_free(exr_default_allocator(), edited[i]); free(edited[i]); }
    exr_reader_close(r);
    mem_sink_free(&sink);
    free(data);
}

/* ---- 4. layout-changing edits are rejected distinctly --------------------- */
static void test_layout_rejected(void) {
    const char *path = "test/unit/regression/leangen-scanline.exr";
    uint8_t *data;
    size_t size;
    exr_reader *r = NULL;
    mem_sink sink;
    exr_header h;
    const exr_header *ptr = &h;
    exr_result rc;

    memset(&sink, 0, sizeof(sink));
    data = read_file(path, &size);
    REQUIRE(data != NULL);
    REQUIRE(open_memory_reader(data, size, &r) == EXR_SUCCESS);

    /* (a) changing compression changes block geometry */
    memset(&h, 0, sizeof(h));
    REQUIRE(exr_header_copy(exr_default_allocator(), &h, exr_reader_part_header(r, 0)) ==
            EXR_SUCCESS);
    h.compression = EXR_COMPRESSION_RLE;
    {
        exr_data_sink sk = {&sink, mem_sink_write, mem_sink_seek, mem_sink_close};
        rc = exr_rewrite_to_sink(r, &ptr, &sk);
    }
    CHECK(rc == EXR_ERROR_LAYOUT_CHANGED);
    exr_header_free(exr_default_allocator(), &h);

    /* (b) changing the data window changes the chunk count */
    memset(&h, 0, sizeof(h));
    REQUIRE(exr_header_copy(exr_default_allocator(), &h, exr_reader_part_header(r, 0)) ==
            EXR_SUCCESS);
    h.data_window.max_y += 1;
    {
        exr_data_sink sk = {&sink, mem_sink_write, mem_sink_seek, mem_sink_close};
        rc = exr_rewrite_to_sink(r, &ptr, &sk);
    }
    CHECK(rc == EXR_ERROR_LAYOUT_CHANGED);
    exr_header_free(exr_default_allocator(), &h);

    /* (c) changing a channel pixel type changes chunk sizes */
    memset(&h, 0, sizeof(h));
    REQUIRE(exr_header_copy(exr_default_allocator(), &h, exr_reader_part_header(r, 0)) ==
            EXR_SUCCESS);
    h.channels[0].pixel_type =
        h.channels[0].pixel_type == EXR_PIXEL_FLOAT ? EXR_PIXEL_HALF
                                                    : EXR_PIXEL_FLOAT;
    {
        exr_data_sink sk = {&sink, mem_sink_write, mem_sink_seek, mem_sink_close};
        rc = exr_rewrite_to_sink(r, &ptr, &sk);
    }
    CHECK(rc == EXR_ERROR_LAYOUT_CHANGED);
    exr_header_free(exr_default_allocator(), &h);

    exr_reader_close(r);
    mem_sink_free(&sink);
    free(data);
}

/* tiled layout change: tile descriptor edit must be rejected */
static void test_tiled_layout_rejected(void) {
    const char *path = "test/unit/regression/leangen-tiled.exr";
    uint8_t *data;
    size_t size;
    exr_reader *r = NULL;
    mem_sink sink;
    exr_header h;
    const exr_header *ptr = &h;
    exr_result rc;

    memset(&sink, 0, sizeof(sink));
    data = read_file(path, &size);
    REQUIRE(data != NULL);
    REQUIRE(open_memory_reader(data, size, &r) == EXR_SUCCESS);
    memset(&h, 0, sizeof(h));
    REQUIRE(exr_header_copy(exr_default_allocator(), &h, exr_reader_part_header(r, 0)) ==
            EXR_SUCCESS);
    h.tile_x_size += 16;
    {
        exr_data_sink sk = {&sink, mem_sink_write, mem_sink_seek, mem_sink_close};
        rc = exr_rewrite_to_sink(r, &ptr, &sk);
    }
    CHECK(rc == EXR_ERROR_LAYOUT_CHANGED);
    exr_header_free(exr_default_allocator(), &h);
    exr_reader_close(r);
    mem_sink_free(&sink);
    free(data);
}

/* ---- 5. non-seekable sink gives a distinct error -------------------------- */

static void test_non_seekable(void) {
    const char *path = "test/unit/regression/2by2.exr";
    uint8_t *data;
    size_t size;
    exr_reader *r = NULL;
    no_seek_sink ns;
    exr_data_sink sk;
    exr_result rc;

    memset(&ns, 0, sizeof(ns));
    data = read_file(path, &size);
    REQUIRE(data != NULL);
    REQUIRE(open_memory_reader(data, size, &r) == EXR_SUCCESS);
    sk.user = &ns; sk.write = no_seek_write; sk.seek = NULL;
    sk.close = no_seek_close;
    rc = exr_rewrite_to_sink(r, NULL, &sk);
    CHECK(rc == EXR_ERROR_NOT_SEEKABLE);
    CHECK(ns.total == 0 && ns.closes == 0); /* rejected before any I/O */
    exr_reader_close(r);
    free(data);
}

/* ---- 5b. a suspending (non-ready) data source is rejected distinctly ------ */

static void test_blocking_source(void) {
    const char *path = "test/unit/regression/2by2.exr";
    uint8_t *data;
    size_t size;
    exr_reader *r = NULL;
    mem_sink sink;
    exr_data_source ds;
    exr_data_sink sk;
    exr_result rc;

    memset(&sink, 0, sizeof(sink));
    data = read_file(path, &size);
    REQUIRE(data != NULL);
    ds.user = NULL; ds.read = blocking_source_read; ds.total_size = size;
    REQUIRE(exr_reader_open_source(&ds, NULL, &r) == EXR_SUCCESS);
    /* parse_header itself returns WOULD_BLOCK; the rewrite must map it. */
    sk.user = &sink; sk.write = mem_sink_write; sk.seek = mem_sink_seek;
    sk.close = mem_sink_close;
    rc = exr_rewrite_to_sink(r, NULL, &sk);
    CHECK(rc == EXR_ERROR_UNSUPPORTED);
    CHECK(!EXR_OK(rc)); /* must not masquerade as the success-class code */
    CHECK(sink.size == 0);
    exr_reader_close(r);
    mem_sink_free(&sink);
    free(data);
}

/* ---- 6. file helper + end-to-end pixel equality --------------------------- */

static void test_file_helper_roundtrip(void) {
    static const char *fixtures[] = {
        "test/unit/regression/leangen-scanline.exr",
        "test/unit/regression/leangen-tiled.exr",
        "test/unit/regression/leangen-mipmap.exr",
        "test/unit/regression/leangen-ripmap.exr",
        "test/unit/regression/leangen-multipart.exr",
        "test/unit/regression/leangen-deepscanline.exr",
        "test/unit/regression/leangen-deeptiled.exr",
    };
    size_t i;
    for (i = 0; i < sizeof(fixtures) / sizeof(fixtures[0]); ++i) {
        exr_reader *r = NULL;
        exr_result rc;
        const char *out_path = "build/rewrite_out.exr";
        rc = exr_reader_open_file(fixtures[i], NULL, &r);
        REQUIRE(rc == EXR_SUCCESS);
        rc = exr_rewrite_to_file(r, NULL, out_path);
        CHECK(rc == EXR_SUCCESS);
        exr_reader_close(r);
        if (rc == EXR_SUCCESS) {
            /* Deep parts have no flat images[]; compare via flat fixtures only
             * through the loader, and rely on chunk identity elsewhere. */
            if (strstr(fixtures[i], "deep") == NULL)
                pixels_equal(fixtures[i], out_path);
        }
        remove(out_path);
    }
}

/* ---- 7. a chunk larger than the copy buffer still passes byte-identical --- */

static void test_large_chunk_passthrough(void) {
    /* flaga.exr is large enough that at least one chunk exceeds 64 KiB. */
    passthrough_identity("test/unit/regression/flaga.exr", 0);
    passthrough_identity("test/unit/regression/flaga.exr", 1);
}

/* ---- 8. callback source stays bounded in request size (O(1) blocks) ------ */

static void test_bounded_source_requests(void) {
    const char *path = "test/unit/regression/leangen-mipmap.exr";
    uint8_t *data;
    size_t size;
    exr_reader *r = NULL;
    mem_sink sink;
    buf_source bs;
    exr_data_source ds;
    exr_data_sink sk;
    exr_result rc;

    memset(&sink, 0, sizeof(sink));
    memset(&bs, 0, sizeof(bs));
    data = read_file(path, &size);
    REQUIRE(data != NULL);
    bs.bytes = data; bs.size = size;
    ds.user = &bs; ds.read = buf_source_read; ds.total_size = size;
    REQUIRE(exr_reader_open_source(&ds, NULL, &r) == EXR_SUCCESS);
    REQUIRE(exr_reader_parse_header(r) == EXR_SUCCESS);
    sk.user = &sink; sk.write = mem_sink_write; sk.seek = mem_sink_seek;
    sk.close = mem_sink_close;
    rc = exr_rewrite_to_sink(r, NULL, &sk);
    CHECK(rc == EXR_SUCCESS);
    /* The core must never ask the source for a file-sized buffer: every
     * request is bounded by the 64 KiB copy window (metadata prefetch aside,
     * which is itself capped by the reader's read-ahead window). */
    CHECK(bs.max_request <= 65536);
    CHECK(sink.size > 0);
    exr_reader_close(r);
    mem_sink_free(&sink);
    free(data);
}

int main(void) {
    printf("== header-only rewrite: passthrough identity ==\n");
    test_passthrough_all_layouts();
    printf("== header-only rewrite: attribute edit ==\n");
    test_attribute_edit();
    printf("== header-only rewrite: multipart edit ==\n");
    test_multipart_edit();
    printf("== header-only rewrite: layout change rejected ==\n");
    test_layout_rejected();
    test_tiled_layout_rejected();
    printf("== header-only rewrite: non-seekable rejected ==\n");
    test_non_seekable();
    printf("== header-only rewrite: blocking source rejected ==\n");
    test_blocking_source();
    printf("== header-only rewrite: file helper round-trip ==\n");
    test_file_helper_roundtrip();
    printf("== header-only rewrite: large chunk passthrough ==\n");
    test_large_chunk_passthrough();
    printf("== header-only rewrite: bounded source requests ==\n");
    test_bounded_source_requests();

    if (failed) {
        fprintf(stderr, "\ntest_exr_rewrite: %d FAILED, %d passed\n", failed,
                passed);
        return 1;
    }
    printf("\ntest_exr_rewrite: all %d checks passed\n", passed);
    return 0;
}
