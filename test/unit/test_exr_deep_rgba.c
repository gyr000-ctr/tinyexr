/*
 * TinyEXR v3 - deep part -> RGBA ("over") compositing test.
 *
 * Build + run (from the repository root, after `make lib`):
 *     cc -O1 -std=c11 -Iinclude test/unit/test_exr_deep_rgba.c \
 *        build/libtinyexr3.a -lm -o /tmp/t && /tmp/t
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "exr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failed, passed;
#define CHECK(c) do {                                                          \
    if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
                ++failed; } else ++passed;                                     \
} while (0)

static exr_channel *make_channels(int n) {
    exr_channel *c = (exr_channel *)calloc((size_t)n, sizeof(*c));
    int i;
    for (i = 0; i < n; ++i) {
        c[i].pixel_type = EXR_PIXEL_FLOAT;
        c[i].x_sampling = c[i].y_sampling = 1;
    }
    return c;
}

static void set_name(exr_channel *c, const char *n) {
    memset(c->name, 0, sizeof(c->name));
    strncpy(c->name, n, sizeof(c->name) - 1);
}

/* 2x1 image, two samples per pixel. Feeds FLOAT planes per channel. */
static void build_part(exr_part *p, exr_channel *ch, int nch, void **planes,
                       int32_t *counts, uint64_t total) {
    memset(p, 0, sizeof(*p));
    p->header.num_channels = nch;
    p->header.channels = ch;
    p->width = 2;
    p->height = 1;
    p->is_deep = 1;
    p->deep_sample_counts = counts;
    p->deep_images = planes;
    p->deep_total_samples = total;
}

/* Front-to-back over, depth sorted: nearest sample wins where alpha is opaque,
 * and the far sample is attenuated by the transmittance of the near one. */
static void basic_composite(void) {
    exr_channel *ch = make_channels(5);
    float R[4] = {1.0f, 0.0f,  0.0f, 0.0f};   /* pixel 0: near red, far green */
    float G[4] = {0.0f, 1.0f,  0.0f, 0.0f};
    float B[4] = {0.0f, 0.0f,  0.0f, 0.0f};
    float A[4] = {0.5f, 0.5f,  0.0f, 0.0f};
    float Z[4] = {2.0f, 1.0f,  0.0f, 0.0f};   /* pixel 0 arrives far-first */
    void *planes[5];
    int32_t counts[2] = {2, 2};
    exr_part p;
    float *rgba = NULL;
    int w = 0, h = 0, oc = 0;
    exr_result rc;

    set_name(&ch[0], "R"); set_name(&ch[1], "G"); set_name(&ch[2], "B");
    set_name(&ch[3], "A"); set_name(&ch[4], "Z");
    planes[0] = R; planes[1] = G; planes[2] = B; planes[3] = A; planes[4] = Z;
    build_part(&p, ch, 5, planes, counts, 4);

    rc = exr_part_to_rgba_float(NULL, &p, &rgba, &w, &h, &oc);
    CHECK(rc == EXR_SUCCESS);
    CHECK(rgba != NULL && w == 2 && h == 1 && oc == 4);
    if (rc == EXR_SUCCESS && rgba) {
        /* Z sorts pixel 0 near-first (the far sample is file index 0): the
         * near sample (G, A=.5) contributes at T=1, then the far one (R, A=.5)
         * is attenuated by T=.5. Sample RGB is premultiplied, so it is used
         * as-is. */
        CHECK(rgba[0] == 0.5f);            /* far red * T(0.5) */
        CHECK(rgba[1] == 1.0f);            /* near green at full transmittance */
        CHECK(rgba[2] == 0.0f);
        CHECK(rgba[3] == 0.75f);           /* 0.5 + 0.5*0.5 */
        /* Pixel 1 has zero-alpha samples: contributes nothing. */
        CHECK(rgba[4] == 0.0f && rgba[5] == 0.0f && rgba[6] == 0.0f &&
              rgba[7] == 0.0f);
    }
    free(rgba);

    /* Out-params are optional, and *out is NULL when the call fails. */
    rgba = (float *)1;
    rc = exr_part_to_rgba_float(NULL, &p, &rgba, NULL, NULL, NULL);
    CHECK(rc == EXR_SUCCESS && rgba != NULL && rgba != (float *)1);
    free(rgba);
    free(ch);
}

/* Without Z, samples composite in file order (OpenEXR's documented fallback). */
static void no_z_file_order(void) {
    exr_channel *ch = make_channels(4);
    float R[2] = {1.0f, 0.0f};
    float G[2] = {0.0f, 1.0f};
    float B[2] = {0.0f, 0.0f};
    float A[2] = {1.0f, 1.0f};
    void *planes[4];
    int32_t counts[2] = {2, 0}; /* both samples in pixel 0; pixel 1 is empty */
    exr_part p;
    float *rgba = NULL;
    exr_result rc;

    set_name(&ch[0], "R"); set_name(&ch[1], "G"); set_name(&ch[2], "B");
    set_name(&ch[3], "A");
    planes[0] = R; planes[1] = G; planes[2] = B; planes[3] = A;
    build_part(&p, ch, 4, planes, counts, 2);

    rc = exr_part_to_rgba_float(NULL, &p, &rgba, NULL, NULL, NULL);
    CHECK(rc == EXR_SUCCESS && rgba != NULL);
    if (rc == EXR_SUCCESS && rgba) {
        /* Pixel 0 composites white then green in file order; the second sample
         * is behind an opaque first one and is fully occluded. */
        CHECK(rgba[0] == 1.0f && rgba[1] == 0.0f && rgba[3] == 1.0f);
        CHECK(rgba[4] == 0.0f && rgba[5] == 0.0f && rgba[7] == 0.0f);
    }
    free(rgba);
    free(ch);
}

/* Missing or non-full-resolution required channels are UNSUPPORTED_CHANNEL,
 * never INVALID_ARGUMENT; a bad part layout is INVALID_ARGUMENT. */
static void error_classes(void) {
    exr_channel *ch = make_channels(4);
    float R[2] = {1.0f, 1.0f}, G[2] = {0.0f, 0.0f}, B[2] = {0.0f, 0.0f};
    float A[2] = {1.0f, 1.0f};
    void *planes[4];
    int32_t counts[2] = {1, 1};
    exr_part p;
    float *rgba = NULL;

    set_name(&ch[0], "R"); set_name(&ch[1], "G"); set_name(&ch[2], "B");
    set_name(&ch[3], "A");
    planes[0] = R; planes[1] = G; planes[2] = B; planes[3] = A;
    build_part(&p, ch, 4, planes, counts, 2);

    /* Missing alpha. */
    {
        exr_channel *c3 = make_channels(3);
        void *p3[3] = {R, G, B};
        exr_part q = p;
        set_name(&c3[0], "R"); set_name(&c3[1], "G"); set_name(&c3[2], "B");
        q.header.channels = c3;
        q.header.num_channels = 3;
        q.deep_images = p3;
        rgba = (float *)1;
        CHECK(exr_part_to_rgba_float(NULL, &q, &rgba, NULL, NULL, NULL) ==
              EXR_ERROR_UNSUPPORTED_CHANNEL);
        CHECK(rgba == NULL);
        free(c3);
    }

    /* A subsampled required channel cannot be expressed as one sample/pixel. */
    ch[1].x_sampling = 2;
    CHECK(exr_part_to_rgba_float(NULL, &p, &rgba, NULL, NULL, NULL) ==
          EXR_ERROR_UNSUPPORTED_CHANNEL && rgba == NULL);
    ch[1].x_sampling = 1;

    /* Structurally bad part: negative sample count. */
    counts[0] = -1;
    CHECK(exr_part_to_rgba_float(NULL, &p, &rgba, NULL, NULL, NULL) ==
          EXR_ERROR_INVALID_ARGUMENT && rgba == NULL);
    counts[0] = 1;

    /* Structurally bad part: deep_total_samples disagrees with the counts. */
    p.deep_total_samples = 99;
    CHECK(exr_part_to_rgba_float(NULL, &p, &rgba, NULL, NULL, NULL) ==
          EXR_ERROR_INVALID_ARGUMENT && rgba == NULL);
    p.deep_total_samples = 2;

    /* Structurally bad part: no per-pixel counts at all. */
    p.deep_sample_counts = NULL;
    CHECK(exr_part_to_rgba_float(NULL, &p, &rgba, NULL, NULL, NULL) ==
          EXR_ERROR_INVALID_ARGUMENT && rgba == NULL);
    p.deep_sample_counts = counts;

    /* Null out pointer is an argument error, not an unsupported layout. */
    CHECK(exr_part_to_rgba_float(NULL, &p, NULL, NULL, NULL, NULL) ==
          EXR_ERROR_INVALID_ARGUMENT);

    /* The good part still succeeds afterwards (no state leaks). */
    CHECK(exr_part_to_rgba_float(NULL, &p, &rgba, NULL, NULL, NULL) ==
          EXR_SUCCESS && rgba != NULL && rgba[3] == 1.0f);
    free(rgba);
    free(ch);
}

/* Empty deep pixels (count 0) and an all-zero image produce transparent black,
 * and every supported source pixel type widens correctly. */
static void mixed_types_and_empty(void) {
    exr_channel *ch = make_channels(5);
    uint16_t Rh[3], Ah[3];
    float Gf[3] = {0.5f, 0.0f, 0.0f};
    float Bf[3] = {0.0f, 0.0f, 0.0f};
    uint32_t Zf[3] = {1u, 2u, 3u};
    void *planes[5];
    int32_t counts[3] = {1, 0, 2};
    exr_part p;
    float *rgba = NULL;
    exr_result rc;
    int i;

    /* HALF R/A, FLOAT G/B, UINT Z. */
    ch[0].pixel_type = EXR_PIXEL_HALF;
    ch[3].pixel_type = EXR_PIXEL_HALF;
    ch[4].pixel_type = EXR_PIXEL_UINT;
    set_name(&ch[0], "R"); set_name(&ch[1], "G"); set_name(&ch[2], "B");
    set_name(&ch[3], "A"); set_name(&ch[4], "Z");
    planes[0] = Rh; planes[1] = Gf; planes[2] = Bf; planes[3] = Ah;
    planes[4] = Zf;
    for (i = 0; i < 3; ++i) { Rh[i] = 0x3c00; Ah[i] = 0x3800; } /* 1.0, 0.5 */

    memset(&p, 0, sizeof(p));
    p.header.num_channels = 5;
    p.header.channels = ch;
    p.width = 3;
    p.height = 1;
    p.is_deep = 1;
    p.deep_sample_counts = counts;
    p.deep_images = planes;
    p.deep_total_samples = 3;

    rc = exr_part_to_rgba_float(NULL, &p, &rgba, NULL, NULL, NULL);
    CHECK(rc == EXR_SUCCESS && rgba != NULL);
    if (rc == EXR_SUCCESS && rgba) {
        /* Premultiplied sample: R channel is HALF 1.0 and A is HALF 0.5, so the
         * pixel is (1.0, 0.5, 0, 0.5) carried straight through. */
        CHECK(rgba[0] == 1.0f && rgba[1] == 0.5f && rgba[3] == 0.5f);
        /* Empty pixel: transparent black. */
        CHECK(rgba[4] == 0.0f && rgba[5] == 0.0f && rgba[6] == 0.0f &&
              rgba[7] == 0.0f);
        /* Pixel 2: two equal-distance samples; first is opaque (half 0.5*
         * ... both have alpha .5) so we get both attenuated. */
        CHECK(rgba[8] > 0.0f && rgba[11] > 0.0f);
    }
    free(rgba);
    free(ch);
}

/* A caller-supplied allocator must receive every allocation back. */
typedef struct stats { size_t live, allocs; } stats;
static void *count_alloc(void *u, size_t n) {
    stats *s = (stats *)u;
    void *p = malloc(n ? n : 1);
    if (p) ++s->live, ++s->allocs;
    return p;
}
static void count_free(void *u, void *p) {
    stats *s = (stats *)u;
    if (p) --s->live;
    free(p);
}
static void allocator_balance(void) {
    exr_channel *ch = make_channels(4);
    float R[2] = {0.5f, 0.5f}, G[2] = {0.0f, 0.0f};
    float B[2] = {0.0f, 0.0f}, A[2] = {0.5f, 0.5f};
    void *planes[4] = {R, G, B, A};
    int32_t counts[2] = {1, 1};
    exr_part p;
    exr_allocator al;
    stats s;
    float *rgba = NULL;

    set_name(&ch[0], "R"); set_name(&ch[1], "G"); set_name(&ch[2], "B");
    set_name(&ch[3], "A");
    build_part(&p, ch, 4, planes, counts, 2);
    memset(&s, 0, sizeof(s));
    al.user = &s; al.alloc = count_alloc; al.free = count_free;
    CHECK(exr_part_to_rgba_float(&al, &p, &rgba, NULL, NULL, NULL) ==
          EXR_SUCCESS && rgba != NULL);
    CHECK(s.live == 1);
    al.free(&s, rgba);
    CHECK(s.live == 0);
    free(ch);
}

int main(void) {
    basic_composite();
    no_z_file_order();
    error_classes();
    mixed_types_and_empty();
    allocator_balance();
    if (failed) {
        fprintf(stderr, "deep_to_rgba: %d failed, %d passed\n", failed, passed);
        return 1;
    }
    printf("deep_to_rgba ok\n");
    return 0;
}
