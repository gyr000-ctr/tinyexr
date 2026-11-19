/*
 * TinyEXR v3 - sanitizer-backed checks for deep -> RGBA compositing.
 *
 * Synthetic parts exercise every guard (hostile counts, missing planes,
 * unsupported layouts); the real deep fixtures confirm a normal image
 * composites and that every returned buffer is released cleanly (LSan).
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

static void *counting_alloc(void *u, size_t n) {
    size_t *live = (size_t *)u;
    void *p = malloc(n ? n : 1);
    if (p) ++*live;
    return p;
}
static void counting_free(void *u, void *p) {
    size_t *live = (size_t *)u;
    if (p) --*live;
    free(p);
}

/* Hostile / inconsistent plumbing must be rejected without touching memory. */
static void hostile_parts(void) {
    exr_channel ch[4];
    float R[2] = {1.0f, 1.0f}, G[2] = {0.0f, 0.0f}, B[2] = {0.0f, 0.0f};
    float A[2] = {1.0f, 1.0f};
    void *planes[4] = {R, G, B, A};
    exr_part p;
    float *rgba = NULL;
    int i;

    memset(ch, 0, sizeof(ch));
    for (i = 0; i < 4; ++i) {
        ch[i].pixel_type = EXR_PIXEL_FLOAT;
        ch[i].x_sampling = ch[i].y_sampling = 1;
    }
    strcpy(ch[0].name, "R"); strcpy(ch[1].name, "G");
    strcpy(ch[2].name, "B"); strcpy(ch[3].name, "A");

    memset(&p, 0, sizeof(p));
    p.header.num_channels = 4;
    p.header.channels = ch;
    p.width = 2;
    p.height = 1;
    p.is_deep = 1;
    p.deep_images = planes;

    /* Counts absent but a positive total recorded. */
    p.deep_total_samples = 0;
    {
        int32_t counts[2] = {1, -5};
        p.deep_sample_counts = counts;
        CHECK(exr_part_to_rgba_float(NULL, &p, &rgba, NULL, NULL, NULL) ==
              EXR_ERROR_INVALID_ARGUMENT && !rgba);
    }
    /* An unset total (0) is harmless, but a total that contradicts the counts
     * is refused. Neither value is ever used to index a plane. */
    {
        int32_t counts[2] = {1, 1};
        p.deep_sample_counts = counts;
        p.deep_total_samples = 0;
        CHECK(exr_part_to_rgba_float(NULL, &p, &rgba, NULL, NULL, NULL) ==
              EXR_SUCCESS && rgba);
        free(rgba);
        rgba = NULL;
        p.deep_total_samples = 7;
        CHECK(exr_part_to_rgba_float(NULL, &p, &rgba, NULL, NULL, NULL) ==
              EXR_ERROR_INVALID_ARGUMENT && !rgba);
        p.deep_total_samples = 2;
    }
    /* Counts that overflow the sample total. */
    {
        int32_t counts[2] = {2147483647, 2147483647};
        p.deep_sample_counts = counts;
        CHECK(exr_part_to_rgba_float(NULL, &p, &rgba, NULL, NULL, NULL) !=
              EXR_SUCCESS && !rgba);
    }
    /* Plane pointer missing while samples are promised. */
    {
        int32_t counts[2] = {1, 1};
        void *missing[4] = {R, G, NULL, A};
        p.deep_sample_counts = counts;
        p.deep_total_samples = 2;
        p.deep_images = missing;
        CHECK(exr_part_to_rgba_float(NULL, &p, &rgba, NULL, NULL, NULL) ==
              EXR_ERROR_INVALID_ARGUMENT && !rgba);
        p.deep_images = planes;
    }
    /* Zero-sized part. */
    {
        int32_t counts[2] = {1, 1};
        p.deep_sample_counts = counts;
        p.deep_total_samples = 2;
        p.width = 0;
        CHECK(exr_part_to_rgba_float(NULL, &p, &rgba, NULL, NULL, NULL) ==
              EXR_ERROR_INVALID_ARGUMENT && !rgba);
        p.width = 2;
    }
    /* A non-deep part still takes the original gather path (zero regression). */
    p.is_deep = 0;
    p.images = planes;
    CHECK(exr_part_to_rgba_float(NULL, &p, &rgba, NULL, NULL, NULL) ==
          EXR_SUCCESS && rgba != NULL);
    free(rgba);
}

/* Composite the bundled deep fixtures and release through a counting allocator
 * so LSan also sees a balanced heap for the library-side scratch. */
static void fixture_composite(const char *path) {
    exr_image img;
    float *rgba = NULL;
    exr_allocator al;
    size_t live = 0;
    int w = 0, h = 0, oc = 0;
    exr_result rc;
    size_t i, expected;

    memset(&img, 0, sizeof(img));
    rc = exr_load_from_file(path, NULL, &img);
    if (!EXR_OK(rc)) { CHECK(0); return; }
    CHECK(img.num_parts == 1 && img.parts[0].is_deep);

    al.user = &live; al.alloc = counting_alloc; al.free = counting_free;
    rc = exr_part_to_rgba_float(&al, &img.parts[0], &rgba, &w, &h, &oc);
    CHECK((rc == EXR_SUCCESS && rgba && oc == 4 && w > 0 && h > 0) ||
          rc == EXR_ERROR_UNSUPPORTED_CHANNEL);
    if (rc == EXR_SUCCESS && rgba) {
        expected = (size_t)w * (size_t)h * 4u;
        /* Alpha is a partial-coverage sum, so it stays within [0, 1]. */
        for (i = 0; i < expected; i += 4)
            if (!(rgba[i + 3] >= 0.0f && rgba[i + 3] <= 1.0f)) break;
        CHECK(i >= expected);
        al.free(&live, rgba);
    }
    CHECK(live == 0);
    exr_image_free(&img);
}

int main(void) {
    hostile_parts();
    fixture_composite("data/deepscanline.exr");
    fixture_composite("data/deep_tiled_sample.exr");
    if (failed) {
        fprintf(stderr, "deep_to_rgba_san: %d failed, %d passed\n", failed,
                passed);
        return 1;
    }
    printf("deep_to_rgba_san ok\n");
    return 0;
}
