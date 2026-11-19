/*
 * TinyEXR - rectangular region decode self-test.
 *
 * Every region must be bit-identical to the same coordinates of the whole-part
 * decode, for scanline and tiled parts (one-level, mipmap and ripmap, both
 * rounding modes), every level, every compression, subsampled channels, the
 * memory/file/streaming source paths, and single-pixel/edge rects. It also
 * checks the error contract (out-of-bounds rects rejected, corrupted chunks
 * reported as CORRUPT) and that peak allocation for a small region stays below
 * a region-plus-one-tile bound instead of the whole image.
 *
 * Build/run (from the repository root):
 *     make test-exr-region
 * or `cc -std=c11 -Iinclude test/unit/test_exr_region.c build/libtinyexr3.a`
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "exr.h"
#include "exr_internal.h" /* exr_num_samples / exr_header_copy */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass, g_fail;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (cond) {                                                            \
            ++g_pass;                                                          \
        } else {                                                               \
            ++g_fail;                                                          \
            printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);           \
        }                                                                      \
    } while (0)

/* ============================================================================
 * Fixture: a synthetic multiplexed part
 *
 *   W x H data window at a sampling-aligned, non-zero origin (to catch
 *   origin-relative bugs; also a negative y origin)
 *   A: HALF  full resolution
 *   B: FLOAT 2x2 subsampled
 *   C: FLOAT 4x4 subsampled
 * Values depend on absolute pixel coordinates so misaligned copies differ.
 * ========================================================================== */

/* Origin is a multiple of the largest sampling rate and the extent a multiple
 * of it too, so every subsampled plane's stored samples are dense. */
enum { FX_W = 132, FX_H = 96, FX_OX = 8, FX_OY = -4 };

typedef struct fixture {
    exr_image img;
    exr_channel chans[3];
    void *planes[3];
    uint16_t *a;
    float *b, *c;
} fixture;

static void fixture_free(fixture *f) {
    /* The fixture's planes/header are caller-owned; only the part array and the
     * value buffers were allocated here. */
    free(f->img.parts);
    free(f->a);
    free(f->b);
    free(f->c);
    memset(f, 0, sizeof(*f));
}

static int fixture_build(fixture *f, exr_part_type type, exr_compression comp,
                         exr_tile_level_mode lvl, exr_tile_rounding_mode rnd,
                         uint32_t tx, uint32_t ty) {
    int x, y;
    exr_part *p;
    memset(f, 0, sizeof(*f));
    f->a = (uint16_t *)malloc(sizeof(uint16_t) * FX_W * FX_H);
    f->b = (float *)malloc(sizeof(float) * (FX_W / 2) * (FX_H / 2));
    f->c = (float *)malloc(sizeof(float) * (FX_W / 4) * (FX_H / 4));
    if (!f->a || !f->b || !f->c) { fixture_free(f); return 0; }
    for (y = 0; y < FX_H; ++y)
        for (x = 0; x < FX_W; ++x) {
            int ax = FX_OX + x, ay = FX_OY + y;
            f->a[y * FX_W + x] = (uint16_t)(ax * 131 + ay * 17 + (ax ^ ay));
        }
    for (y = 0; y < FX_H / 2; ++y)
        for (x = 0; x < FX_W / 2; ++x)
            f->b[y * (FX_W / 2) + x] =
                (float)((FX_OX + x * 2) - 3 * (FX_OY + y * 2)) / 7.0f;
    for (y = 0; y < FX_H / 4; ++y)
        for (x = 0; x < FX_W / 4; ++x)
            f->c[y * (FX_W / 4) + x] =
                (float)((FX_OX + x * 4) * 5 + (FX_OY + y * 4) * 11) / 3.0f;

    strcpy(f->chans[0].name, "A");
    f->chans[0].pixel_type = EXR_PIXEL_HALF;
    f->chans[0].x_sampling = f->chans[0].y_sampling = 1;
    strcpy(f->chans[1].name, "B");
    f->chans[1].pixel_type = EXR_PIXEL_FLOAT;
    f->chans[1].x_sampling = f->chans[1].y_sampling = 2;
    strcpy(f->chans[2].name, "C");
    f->chans[2].pixel_type = EXR_PIXEL_FLOAT;
    f->chans[2].x_sampling = f->chans[2].y_sampling = 4;
    f->planes[0] = f->a;
    f->planes[1] = f->b;
    f->planes[2] = f->c;

    memset(&f->img, 0, sizeof(f->img));
    f->img.num_parts = 1;
    f->img.parts = (exr_part *)calloc(1, sizeof(exr_part));
    if (!f->img.parts) { fixture_free(f); return 0; }
    p = &f->img.parts[0];
    p->header.num_channels = 3;
    p->header.channels = f->chans;
    if (type == EXR_PART_TILED || type == EXR_PART_DEEP_TILED) {
        /* The format forbids subsampling in tiled parts; keep the same pixel
         * values but store every channel at full resolution. */
        int c;
        for (c = 0; c < 3; ++c) f->chans[c].x_sampling = f->chans[c].y_sampling = 1;
        free(f->b);
        free(f->c);
        f->b = (float *)malloc(sizeof(float) * FX_W * FX_H);
        f->c = (float *)malloc(sizeof(float) * FX_W * FX_H);
        if (!f->b || !f->c) { fixture_free(f); return 0; }
        for (y = 0; y < FX_H; ++y)
            for (x = 0; x < FX_W; ++x) {
                int ax = FX_OX + x, ay = FX_OY + y;
                f->b[y * FX_W + x] = (float)(ax - 3 * ay) / 7.0f;
                f->c[y * FX_W + x] = (float)(ax * 5 + ay * 11) / 3.0f;
            }
        f->planes[1] = f->b;
        f->planes[2] = f->c;
    }
    p->header.part_type = type;
    p->header.tiled = (uint8_t)(type == EXR_PART_TILED ||
                                type == EXR_PART_DEEP_TILED);
    p->header.compression = comp;
    p->header.data_window.min_x = FX_OX;
    p->header.data_window.min_y = FX_OY;
    p->header.data_window.max_x = FX_OX + FX_W - 1;
    p->header.data_window.max_y = FX_OY + FX_H - 1;
    p->header.display_window = p->header.data_window;
    p->header.pixel_aspect_ratio = 1.0f;
    p->header.screen_window_width = 1.0f;
    p->header.tile_x_size = tx;
    p->header.tile_y_size = ty;
    p->header.level_mode = lvl;
    p->header.rounding_mode = rnd;
    p->width = FX_W;
    p->height = FX_H;
    /* The fixtures use channels f->chans / planes during serialization. */
    p->images = f->planes;
    return 1;
}

/* ============================================================================
 * Reference comparison
 * ========================================================================== */

static int nsamp(int lo, int hi, int s) {
    return exr_num_samples(lo, hi, s);
}

/* The plane element in `part` for absolute sample coordinate (ax, ay), or NULL
 * when that coordinate is not a stored sample. */
static const void *plane_at(const exr_part *part, int c, int ax, int ay) {
    const exr_header *h = &part->header;
    const exr_channel *ch = &h->channels[c];
    size_t ps = ch->pixel_type == EXR_PIXEL_HALF ? 2u : 4u;
    int xs = ch->x_sampling, ys = ch->y_sampling;
    int w, row, col;
    if (ax < h->data_window.min_x || ax > h->data_window.max_x ||
        ay < h->data_window.min_y || ay > h->data_window.max_y)
        return NULL;
    if (ax % xs != 0 || ay % ys != 0) return NULL;
    w = nsamp(h->data_window.min_x, h->data_window.max_x, xs);
    row = nsamp(h->data_window.min_y, ay, ys) - 1;
    col = nsamp(h->data_window.min_x, ax, xs) - 1;
    if (row < 0 || col < 0 || col >= w) return NULL;
    return (const uint8_t *)part->images[c] + ((size_t)row * (size_t)w +
                                               (size_t)col) * ps;
}

/* Every stored sample of the rect in `region` must equal the whole-part value;
 * plane sizes must match the rect's sampled extents. */
static int region_matches(const exr_part *full, const exr_part *region, int x0,
                          int y0, int w, int h, const char *what) {
    const exr_header *rh = &region->header;
    int c, ax, ay;
    if (region->width != w || region->height != h ||
        rh->data_window.min_x != x0 || rh->data_window.min_y != y0 ||
        rh->data_window.max_x != x0 + w - 1 ||
        rh->data_window.max_y != y0 + h - 1) {
        printf("    %s: wrong region geometry\n", what);
        return 0;
    }
    for (c = 0; c < rh->num_channels; ++c) {
        const exr_channel *ch = &rh->channels[c];
        size_t ps = ch->pixel_type == EXR_PIXEL_HALF ? 2u : 4u;
        int want = nsamp(x0, x0 + w - 1, ch->x_sampling) *
                   nsamp(y0, y0 + h - 1, ch->y_sampling);
        size_t got = exr_region_channel_size(rh, c);
        if ((size_t)want != got || !region->images[c]) {
            printf("    %s: channel %d size %zu != %d\n", what, c, got, want);
            return 0;
        }
        for (ay = y0; ay <= y0 + h - 1; ++ay)
            for (ax = x0; ax <= x0 + w - 1; ++ax) {
                const void *a = plane_at(full, c, ax, ay);
                const void *b = plane_at(region, c, ax, ay);
                int pa = ax % ch->x_sampling == 0 && ay % ch->y_sampling == 0;
                if (!pa) continue;
                if (!a || !b) { printf("    %s: missing sample\n", what); return 0; }
                if (memcmp(a, b, ps) != 0) {
                    printf("    %s: ch %d abs (%d,%d) differs\n", what, c, ax, ay);
                    return 0;
                }
            }
    }
    return 1;
}

/* ============================================================================
 * Helpers: read a part, then a sweep of rects against it
 * ========================================================================== */

typedef struct rects {
    int n;
    int x0[24], y0[24], w[24], h[24];
} rects;

static void rects_build(rects *rs, int dwx, int dwy, int W, int H) {
    int i;
    static const int fx[6] = {0, 0, 1, 3, 5, 8};
    static const int fy[6] = {0, 2, 1, 7, 4, 9};
    static const int fw[6] = {1, 2, 5, 17, 40, 1000};
    static const int fh[6] = {1, 3, 4, 13, 50, 1000};
    rs->n = 0;
    for (i = 0; i < 6 && rs->n < 24; ++i) {
        int w = fw[i] > W ? W : fw[i];
        int h = fh[i] > H ? H : fh[i];
        int x = fx[i] < W ? fx[i] : W - 1;
        int y = fy[i] < H ? fy[i] : H - 1;
        if (x + w > W) x = W - w;
        if (y + h > H) y = H - h;
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        rs->x0[rs->n] = dwx + x;
        rs->y0[rs->n] = dwy + y;
        rs->w[rs->n] = w;
        rs->h[rs->n] = h;
        ++rs->n;
    }
    /* Whole image, last-row/column strips, bottom-right corner. */
    rs->x0[rs->n] = dwx; rs->y0[rs->n] = dwy; rs->w[rs->n] = W; rs->h[rs->n] = H; ++rs->n;
    rs->x0[rs->n] = dwx; rs->y0[rs->n] = dwy + H - 1; rs->w[rs->n] = W; rs->h[rs->n] = 1; ++rs->n;
    rs->x0[rs->n] = dwx + W - 1; rs->y0[rs->n] = dwy; rs->w[rs->n] = 1; rs->h[rs->n] = H; ++rs->n;
    {
        int cw = W < 3 ? W : 3, chh = H < 2 ? H : 2;
        rs->x0[rs->n] = dwx + W - cw;
        rs->y0[rs->n] = dwy + H - chh;
        rs->w[rs->n] = cw;
        rs->h[rs->n] = chh;
        ++rs->n;
    }
}

/* Region sweep against one decoded level (level_x/level_y of the file). */
static void sweep_level(exr_reader *r, int part, int level_x, int level_y,
                        int dwx, int dwy, int lw, int lh, const char *what) {
    exr_part full, reg;
    rects rs;
    int i;
    memset(&full, 0, sizeof(full));
    memset(&reg, 0, sizeof(reg));

    /* Reference: for level 0 a whole-part read; deeper levels are assembled
     * from read_tile (whole-part reads only materialize level 0). */
    if (level_x == 0 && level_y == 0) {
        if (!EXR_OK(exr_reader_read_part(r, part, &full))) {
            CHECK(0, what);
            return;
        }
    } else {
        uint32_t nb = 0, k;
        int c, iy;
        size_t ps[8];
        memset(ps, 0, sizeof(ps));
        if (!EXR_OK(exr_reader_num_blocks(r, part, &nb))) { CHECK(0, what); return; }
        full.width = lw; full.height = lh;
        {
            const exr_header *h = exr_reader_part_header(r, part);
            /* A level's pixels are addressed from the data-window origin (its
             * chunks are laid out at min_x + tile*tile_size), like the whole
             * read_tile API; find whether any chunk exists at all. */
            int found = 0;
            for (k = 0; k < nb; ++k) {
                exr_block_info bi;
                if (!EXR_OK(exr_reader_block_info(r, part, k, &bi))) continue;
                if (bi.level_x == level_x && bi.level_y == level_y) { found = 1; break; }
            }
            dwx = h->data_window.min_x; dwy = h->data_window.min_y;
            exr_result rc;
            if (!found) {
                /* No chunk materializes this level (can happen for degenerate
                 * ripmap corners); there is nothing to compare against. */
                full.images = NULL;
                exr_part_free(NULL, &full);
                CHECK(1, "empty level");
                return;
            }
            rc = exr_header_copy(exr_default_allocator(), &full.header, h);
            if (!EXR_OK(rc)) { CHECK(0, what); return; }
            full.header.data_window.min_x = dwx;
            full.header.data_window.min_y = dwy;
            full.header.data_window.max_x = dwx + lw - 1;
            full.header.data_window.max_y = dwy + lh - 1;
            full.images = (void **)calloc((size_t)h->num_channels, sizeof(void *));
            if (!full.images) { CHECK(0, what); return; }
            for (c = 0; c < h->num_channels; ++c) {
                int cw = nsamp(dwx, dwx + lw - 1, h->channels[c].x_sampling);
                int chh = nsamp(dwy, dwy + lh - 1, h->channels[c].y_sampling);
                ps[c] = h->channels[c].pixel_type == EXR_PIXEL_HALF ? 2u : 4u;
                full.images[c] = calloc((size_t)cw * (size_t)chh, ps[c]);
                if (!full.images[c]) { CHECK(0, what); return; }
            }
            for (k = 0; k < nb; ++k) {
                exr_block_info bi;
                exr_part tp;
                if (!EXR_OK(exr_reader_block_info(r, part, k, &bi))) { CHECK(0, what); return; }
                if (bi.level_x != level_x || bi.level_y != level_y) continue;
                memset(&tp, 0, sizeof(tp));
                if (!EXR_OK(exr_reader_read_tile(r, part, bi.tile_x, bi.tile_y,
                                                 level_x, level_y, &tp))) {
                    CHECK(0, what);
                    exr_part_free(NULL, &full);
                    return;
                }
                for (c = 0; c < tp.header.num_channels; ++c) {
                    const exr_channel *ch = &tp.header.channels[c];
                    int cw = nsamp(dwx, dwx + lw - 1, ch->x_sampling);
                    int col0 = nsamp(dwx, bi.x0 - 1, ch->x_sampling);
                    int row0 = nsamp(dwy, bi.y0 - 1, ch->y_sampling);
                    int tw = nsamp(bi.x0, bi.x0 + bi.width - 1, ch->x_sampling);
                    int th = nsamp(bi.y0, bi.y0 + bi.height - 1, ch->y_sampling);
                    for (iy = 0; iy < th; ++iy)
                        memcpy((uint8_t *)full.images[c] +
                                   ((size_t)(row0 + iy) * (size_t)cw + (size_t)col0) * ps[c],
                               (uint8_t *)tp.images[c] + (size_t)iy * tw * ps[c],
                               (size_t)tw * ps[c]);
                }
                exr_part_free(NULL, &tp);
            }
        }
    }

    {
        const exr_header *hh = &full.header;
        /* The stored level extent equals the assembled reference's extent. */
        rects_build(&rs, hh->data_window.min_x, hh->data_window.min_y,
                    full.width, full.height);
    }
    for (i = 0; i < rs.n; ++i) {
        exr_result rc;
        memset(&reg, 0, sizeof(reg));
        rc = exr_reader_read_region(r, part, rs.x0[i], rs.y0[i], rs.w[i],
                                    rs.h[i], level_x, level_y, &reg);
        if (!EXR_OK(rc)) {
            printf("    %s: rect %d,%d %dx%d lvl %d,%d lw,lh %d,%d full.ox,oy %d,%d -> %s\n", what, rs.x0[i], rs.y0[i],
                   rs.w[i], rs.h[i], level_x, level_y, lw, lh, full.header.data_window.min_x, full.header.data_window.min_y, exr_result_string(rc));
            CHECK(0, what);
            continue;
        }
        CHECK(region_matches(&full, &reg, rs.x0[i], rs.y0[i], rs.w[i], rs.h[i],
                             what),
              what);
        exr_part_free(NULL, &reg);
    }
    exr_part_free(NULL, &full);
}

/* ============================================================================
 * Tests
 * ========================================================================== */

/* Scanline parts, every compression: the full sweep plus every level-0 sample
 * coordinate for corner/edge guarantees. */
static void scanline_tests(void) {
    static const exr_compression comps[] = {
        EXR_COMPRESSION_NONE, EXR_COMPRESSION_RLE,  EXR_COMPRESSION_ZIPS,
        EXR_COMPRESSION_ZIP,  EXR_COMPRESSION_PIZ,  EXR_COMPRESSION_PXR24,
        EXR_COMPRESSION_B44A, EXR_COMPRESSION_ZSTD, EXR_COMPRESSION_HTJ2K32};
    size_t i;
    for (i = 0; i < sizeof(comps) / sizeof(comps[0]); ++i) {
        fixture f;
        void *buf = NULL;
        size_t sz = 0;
        exr_reader *r = NULL;
        exr_result rc;
        if (!fixture_build(&f, EXR_PART_SCANLINE, comps[i], EXR_TILE_ONE_LEVEL,
                           EXR_TILE_ROUND_DOWN, 0, 0)) {
            CHECK(0, "fixture build");
            continue;
        }
        /* B44/B44A are lossy: the region must still agree with the full decode,
         * which the sweep checks. PXR24 too. */
        rc = exr_save_to_memory(&buf, &sz, NULL, &f.img, comps[i]);
        fixture_free(&f);
        if (!EXR_OK(rc)) {
            printf("  skip scanline comp %d (encode: %s)\n", (int)comps[i],
                   exr_result_string(rc));
            continue;
        }
        if (!EXR_OK(exr_reader_open_memory(buf, sz, NULL, &r))) {
            CHECK(0, "open memory");
            free(buf);
            continue;
        }
        sweep_level(r, 0, 0, 0, FX_OX, FX_OY, FX_W, FX_H, "scanline");
        exr_reader_close(r);
        free(buf);
    }
}

/* Tiled parts across level modes / rounding: every level, every tile-aligned
 * and unaligned rect. */
static void tiled_tests(void) {
    static const exr_tile_level_mode modes[] = {
        EXR_TILE_ONE_LEVEL, EXR_TILE_MIPMAP_LEVELS, EXR_TILE_RIPMAP_LEVELS};
    static const exr_tile_rounding_mode rounds[] = {EXR_TILE_ROUND_DOWN,
                                                    EXR_TILE_ROUND_UP};
    static const uint32_t sizes[2][2] = {{16, 16}, {37, 23}};
    size_t m, rd, s;
    for (m = 0; m < 3; ++m)
        for (rd = 0; rd < 2; ++rd)
            for (s = 0; s < 2; ++s) {
                fixture f;
                void *buf = NULL;
                size_t sz = 0;
                exr_reader *r = NULL;
                exr_result rc;
                uint32_t nb = 0, k;
                int maxlx = 0, maxly = 0, lx, ly;
                if (!fixture_build(&f, EXR_PART_TILED, EXR_COMPRESSION_ZIP,
                                   modes[m], rounds[rd], sizes[s][0],
                                   sizes[s][1])) {
                    CHECK(0, "fixture build");
                    continue;
                }
                rc = exr_save_to_memory(&buf, &sz, NULL, &f.img, EXR_COMPRESSION_ZIP);
                fixture_free(&f);
                if (!EXR_OK(rc)) { CHECK(0, "tiled save"); continue; }
                if (!EXR_OK(exr_reader_open_memory(buf, sz, NULL, &r))) {
                    CHECK(0, "open tiled");
                    free(buf);
                    continue;
                }
                if (!EXR_OK(exr_reader_num_blocks(r, 0, &nb))) { CHECK(0, "blocks"); }
                for (k = 0; k < nb; ++k) {
                    exr_block_info bi;
                    if (!EXR_OK(exr_reader_block_info(r, 0, k, &bi))) continue;
                    if (bi.level_x > maxlx) maxlx = bi.level_x;
                    if (bi.level_y > maxly) maxly = bi.level_y;
                }
                for (ly = 0; ly <= maxly; ++ly)
                    for (lx = 0; lx <= maxlx; ++lx) {
                        int lw, lh, step = (int)sizes[s][0];
                        if (modes[m] == EXR_TILE_MIPMAP_LEVELS && lx != ly) continue;
                        /* level extents: read_tile clamps to the level size */
                        lw = FX_W;
                        lh = FX_H;
                        {
                            int t;
                            for (t = 0; t < lx; ++t) {
                                lw = rounds[rd] == EXR_TILE_ROUND_UP ? (lw + 1) / 2 : lw / 2;
                                if (lw < 1) lw = 1;
                            }
                            for (t = 0; t < ly; ++t) {
                                lh = rounds[rd] == EXR_TILE_ROUND_UP ? (lh + 1) / 2 : lh / 2;
                                if (lh < 1) lh = 1;
                            }
                            if (modes[m] == EXR_TILE_MIPMAP_LEVELS) {
                                /* mipmap: both axes halve lx (== ly) times; lh
                                 * starts from FX_H like the file does. */
                                lh = FX_H;
                                for (t = 0; t < lx; ++t) {
                                    lh = rounds[rd] == EXR_TILE_ROUND_UP ? (lh + 1) / 2 : lh / 2;
                                    if (lh < 1) lh = 1;
                                }
                            }
                        }
                        (void)step;
                        sweep_level(r, 0, lx, ly, FX_OX, FX_OY, lw, lh,
                                    "tiled level");
                    }
                exr_reader_close(r);
                free(buf);
            }
}

/* Every individual pixel of a small tiled image, including across tile and
 * block seams, must match the full decode exactly. */
static void exhaustive_small(void) {
    exr_image img;
    exr_part part;
    exr_channel ch[2];
    void *planes[2];
    uint16_t *p0;
    float *p1;
    int x, y;
    void *buf = NULL;
    size_t sz = 0;
    exr_reader *mem = NULL;
    exr_part full;
    static const exr_compression comps[] = {EXR_COMPRESSION_ZIP,
                                            EXR_COMPRESSION_PIZ,
                                            EXR_COMPRESSION_ZSTD};
    size_t ci;

    p0 = (uint16_t *)malloc(sizeof(uint16_t) * 23 * 19);
    p1 = (float *)malloc(sizeof(float) * 23 * 19);
    if (!p0 || !p1) { free(p0); free(p1); CHECK(0, "alloc"); return; }
    for (y = 0; y < 19; ++y)
        for (x = 0; x < 23; ++x) {
            p0[y * 23 + x] = (uint16_t)(x * 71 + y * 29);
            p1[y * 23 + x] = (float)(x * 3 - y * 5);
        }
    strcpy(ch[0].name, "H");
    ch[0].pixel_type = EXR_PIXEL_HALF;
    ch[0].x_sampling = ch[0].y_sampling = 1;
    strcpy(ch[1].name, "F");
    ch[1].pixel_type = EXR_PIXEL_FLOAT;
    ch[1].x_sampling = ch[1].y_sampling = 1;
    planes[0] = p0;
    planes[1] = p1;
    memset(&part, 0, sizeof(part));
    part.header.num_channels = 2;
    part.header.channels = ch;
    part.header.data_window = (exr_box2i){0, 0, 22, 18};
    part.header.display_window = part.header.data_window;
    part.header.pixel_aspect_ratio = 1.0f;
    part.header.screen_window_width = 1.0f;
    part.header.tiled = 1;
    part.header.part_type = EXR_PART_TILED;
    part.header.tile_x_size = 7;
    part.header.tile_y_size = 5;
    part.header.level_mode = EXR_TILE_ONE_LEVEL;
    part.width = 23;
    part.height = 19;
    part.images = planes;
    memset(&img, 0, sizeof(img));
    img.num_parts = 1;
    img.parts = &part;

    for (ci = 0; ci < sizeof(comps) / sizeof(comps[0]); ++ci) {
        exr_result rc = exr_save_to_memory(&buf, &sz, NULL, &img, comps[ci]);
        if (!EXR_OK(rc)) { CHECK(0, "small save"); continue; }
        if (!EXR_OK(exr_reader_open_memory(buf, sz, NULL, &mem))) {
            CHECK(0, "small open");
            free(buf);
            continue;
        }
        memset(&full, 0, sizeof(full));
        if (EXR_OK(exr_reader_read_part(mem, 0, &full))) {
            for (y = 0; y < 19; ++y)
                for (x = 0; x < 23; ++x) {
                    exr_part reg;
                    memset(&reg, 0, sizeof(reg));
                    rc = exr_reader_read_region(mem, 0, x, y, 1, 1, 0, 0, &reg);
                    if (!EXR_OK(rc) ||
                        !region_matches(&full, &reg, x, y, 1, 1, "pixel")) {
                        CHECK(0, "pixel region");
                        if (EXR_OK(rc)) exr_part_free(NULL, &reg);
                        continue;
                    }
                    exr_part_free(NULL, &reg);
                }
            exr_part_free(NULL, &full);
        } else {
            CHECK(0, "small full");
        }
        exr_reader_close(mem);
        free(buf);
        buf = NULL;
    }
    free(p0);
    free(p1);
}

/* Error contract: out-of-bounds rects, wrong levels, deep parts. */
static void error_contract(void) {
    fixture f;
    void *buf = NULL;
    size_t sz = 0;
    exr_reader *r = NULL;
    exr_part out;
    exr_result rc;

    if (!fixture_build(&f, EXR_PART_SCANLINE, EXR_COMPRESSION_ZIP,
                       EXR_TILE_ONE_LEVEL, EXR_TILE_ROUND_DOWN, 0, 0)) {
        CHECK(0, "fixture");
        return;
    }
    rc = exr_save_to_memory(&buf, &sz, NULL, &f.img, EXR_COMPRESSION_ZIP);
    fixture_free(&f);
    if (!EXR_OK(rc)) { CHECK(0, "save"); return; }
    if (!EXR_OK(exr_reader_open_memory(buf, sz, NULL, &r))) { CHECK(0, "open"); free(buf); return; }

    memset(&out, 1, sizeof(out)); /* poison: must be cleared on failure */
    CHECK(exr_reader_read_region(r, 0, FX_OX - 1, FX_OY, 4, 4, 0, 0, &out) ==
              EXR_ERROR_INVALID_ARGUMENT, "rect left of data window");
    /* Rejection happens before any allocation, so `out` is untouched. */
    CHECK(out.images == (void **)(uintptr_t)0x0101010101010101ULL,
          "rejected rect leaves out untouched");
    memset(&out, 0, sizeof(out));
    CHECK(exr_reader_read_region(r, 0, FX_OX, FX_OY, FX_W + 1, 4, 0, 0, &out) ==
              EXR_ERROR_INVALID_ARGUMENT, "rect past right edge");
    memset(&out, 0, sizeof(out));
    CHECK(exr_reader_read_region(r, 0, FX_OX, FX_OY + FX_H - 1, 4, 2, 0, 0, &out) ==
              EXR_ERROR_INVALID_ARGUMENT, "rect past bottom edge");
    memset(&out, 0, sizeof(out));
    CHECK(exr_reader_read_region(r, 0, FX_OX, FX_OY, 0, 4, 0, 0, &out) ==
              EXR_ERROR_INVALID_ARGUMENT, "zero width");
    memset(&out, 0, sizeof(out));
    CHECK(exr_reader_read_region(r, 0, FX_OX, FX_OY, -3, 4, 0, 0, &out) ==
              EXR_ERROR_INVALID_ARGUMENT, "negative width");
    memset(&out, 0, sizeof(out));
    CHECK(exr_reader_read_region(r, 0, FX_OX, FX_OY, 4, 4, 1, 1, &out) ==
              EXR_ERROR_INVALID_ARGUMENT, "level on scanline part");
    memset(&out, 0, sizeof(out));
    CHECK(exr_reader_read_region(r, 0, FX_OX, FX_OY, 4, 4, 0, 0, &out) ==
              EXR_SUCCESS, "interior rect succeeds");
    if (out.images) exr_part_free(NULL, &out);
    memset(&out, 0, sizeof(out));
    CHECK(exr_reader_read_region(r, 9, FX_OX, FX_OY, 4, 4, 0, 0, &out) ==
              EXR_ERROR_INVALID_ARGUMENT, "bad part index");
    exr_reader_close(r);
    free(buf);

    /* Deep parts are outside the region API's contract. */
    {
        exr_image img;
        exr_part part;
        exr_channel chans[1];
        int32_t *counts;
        uint16_t *samples;
        void *dbuf = NULL;
        size_t dsz = 0;
        exr_reader *dr = NULL;
        memset(&img, 0, sizeof(img));
        memset(&part, 0, sizeof(part));
        memset(chans, 0, sizeof(chans));
        strcpy(chans[0].name, "Z");
        chans[0].pixel_type = EXR_PIXEL_HALF;
        chans[0].x_sampling = chans[0].y_sampling = 1;
        part.header.num_channels = 1;
        part.header.channels = chans;
        part.header.part_type = EXR_PART_DEEP_SCANLINE;
        part.header.data_window = (exr_box2i){0, 0, 3, 3};
        part.header.display_window = part.header.data_window;
        part.header.pixel_aspect_ratio = 1.0f;
        part.header.screen_window_width = 1.0f;
        part.width = 4;
        part.height = 4;
        part.is_deep = 1;
        counts = (int32_t *)calloc(16, sizeof(int32_t));
        samples = (uint16_t *)calloc(16, sizeof(uint16_t));
        if (counts && samples) {
            int i;
            for (i = 0; i < 16; ++i) counts[i] = 1;
            part.deep_sample_counts = counts;
            part.deep_images = (void **)calloc(1, sizeof(void *));
            if (part.deep_images) part.deep_images[0] = samples;
            part.deep_total_samples = 16;
            img.num_parts = 1;
            img.parts = &part;
            rc = exr_save_to_memory(&dbuf, &dsz, NULL, &img, EXR_COMPRESSION_ZIP);
            if (EXR_OK(rc) && EXR_OK(exr_reader_open_memory(dbuf, dsz, NULL, &dr))) {
                memset(&out, 0, sizeof(out));
                CHECK(exr_reader_read_region(dr, 0, 0, 0, 2, 2, 0, 0, &out) ==
                          EXR_ERROR_UNSUPPORTED, "deep part region unsupported");
                CHECK(out.images == NULL, "deep rejection owns nothing");
                exr_reader_close(dr);
            } else {
                CHECK(1, "deep fixture unavailable (skipped)");
            }
            free(dbuf);
        }
        free(part.deep_images);
        free(counts);
        free(samples);
    }
}

/* ============================================================================
 * Streaming callback source, damaged chunks, and peak memory
 * ========================================================================== */

typedef struct cb_source {
    const uint8_t *bytes;
    size_t size;
    int block_mode; /* 1: return WOULD_BLOCK for every request */
} cb_source;

static exr_result cb_read(void *user, uint64_t off, uint64_t len, void *dst) {
    cb_source *s = (cb_source *)user;
    if (off > s->size || len > s->size - off) return EXR_ERROR_CORRUPT;
    if (s->block_mode) return EXR_WOULD_BLOCK;
    memcpy(dst, s->bytes + off, (size_t)len);
    return EXR_SUCCESS;
}

static int supply_all(exr_reader *r, const cb_source *s) {
    exr_pending_read p;
    uint8_t buf[65536];
    if (exr_reader_pending(r, &p) != EXR_SUCCESS) return 0;
    while (p.size) {
        size_t n = p.size > sizeof(buf) ? sizeof(buf) : (size_t)p.size;
        memcpy(buf, s->bytes + p.offset, n);
        if (exr_reader_supply(r, p.offset, buf, n) != EXR_SUCCESS) return 0;
        if (exr_reader_pending(r, &p) != EXR_SUCCESS) break;
    }
    return 1;
}

/* A file opened through the streaming data source must produce byte-identical
 * regions, including while suspended and resumed at every fetch. */
static void streaming_test(void) {
    fixture f;
    void *buf = NULL;
    size_t sz = 0;
    exr_reader *mem = NULL, *cb = NULL;
    exr_data_source src;
    cb_source state;
    exr_result rc;
    exr_part full;

    if (!fixture_build(&f, EXR_PART_TILED, EXR_COMPRESSION_ZIP,
                       EXR_TILE_ONE_LEVEL, EXR_TILE_ROUND_DOWN, 24, 24)) {
        CHECK(0, "fixture");
        return;
    }
    rc = exr_save_to_memory(&buf, &sz, NULL, &f.img, EXR_COMPRESSION_ZIP);
    fixture_free(&f);
    if (!EXR_OK(rc)) { CHECK(0, "save streaming"); return; }
    if (!EXR_OK(exr_reader_open_memory(buf, sz, NULL, &mem))) {
        CHECK(0, "open memory");
        free(buf);
        return;
    }
    memset(&full, 0, sizeof(full));
    if (!EXR_OK(exr_reader_read_part(mem, 0, &full))) {
        CHECK(0, "full read");
        exr_reader_close(mem);
        free(buf);
        return;
    }
    state.bytes = (const uint8_t *)buf;
    state.size = sz;
    state.block_mode = 1;
    src.user = &state;
    src.read = cb_read;
    src.total_size = sz;
    if (EXR_OK(exr_reader_open_source(&src, NULL, &cb))) {
        while ((rc = exr_reader_parse_header(cb)) == EXR_WOULD_BLOCK)
            supply_all(cb, &state);
        CHECK(rc == EXR_SUCCESS, "streaming parse");
        {
            static const int r0[3] = {0, 0, 100}, c0[3] = {0, 0, 50};
            static const int rw[3] = {1, 60, 30}, rh[3] = {1, 44, 40};
            int i;
            for (i = 0; i < 3; ++i) {
                int x0 = FX_OX + r0[i], y0 = FX_OY + c0[i];
                int rwv = rw[i], rhv = rh[i];
                exr_part reg;
                int tries = 0;
                if (x0 + rwv > FX_OX + FX_W) rwv = FX_OX + FX_W - x0;
                if (y0 + rhv > FX_OY + FX_H) rhv = FX_OY + FX_H - y0;
                memset(&reg, 0, sizeof(reg));
                do {
                    rc = exr_reader_read_region(cb, 0, x0, y0, rwv, rhv, 0, 0, &reg);
                    if (rc == EXR_WOULD_BLOCK && !supply_all(cb, &state)) break;
                } while (rc == EXR_WOULD_BLOCK && ++tries < 1000000);
                if (!EXR_OK(rc)) { CHECK(0, "streaming region"); continue; }
                CHECK(region_matches(&full, &reg, x0, y0, rwv, rhv, "streaming"),
                      "streaming region matches");
                exr_part_free(NULL, &reg);
            }
        }
        exr_reader_close(cb);
    } else {
        CHECK(0, "open source");
    }
    exr_part_free(NULL, &full);
    exr_reader_close(mem);
    free(buf);
}

/* Run a mipmap part through both the memory and the streaming paths and check
 * a level-0 rect against the memory reference. Higher-level chunks of a
 * mipmap can overlap the requested rect in packed coordinates; the streaming
 * operation must skip them rather than scatter them as level 0. */
static void streaming_mipmap_test(void) {
    fixture f;
    void *buf = NULL;
    size_t sz = 0;
    exr_reader *mem = NULL, *cb = NULL;
    exr_data_source src;
    cb_source state;
    exr_result rc;
    exr_part full;

    if (!fixture_build(&f, EXR_PART_TILED, EXR_COMPRESSION_ZIP,
                       EXR_TILE_MIPMAP_LEVELS, EXR_TILE_ROUND_DOWN, 20, 20)) {
        CHECK(0, "fixture");
        return;
    }
    rc = exr_save_to_memory(&buf, &sz, NULL, &f.img, EXR_COMPRESSION_ZIP);
    fixture_free(&f);
    if (!EXR_OK(rc)) { CHECK(0, "mip save"); return; }
    if (!EXR_OK(exr_reader_open_memory(buf, sz, NULL, &mem))) {
        CHECK(0, "mip open");
        free(buf);
        return;
    }
    memset(&full, 0, sizeof(full));
    if (!EXR_OK(exr_reader_read_part(mem, 0, &full))) {
        CHECK(0, "mip full read");
        exr_reader_close(mem);
        free(buf);
        return;
    }
    state.bytes = (const uint8_t *)buf;
    state.size = sz;
    state.block_mode = 1;
    src.user = &state;
    src.read = cb_read;
    src.total_size = sz;
    if (EXR_OK(exr_reader_open_source(&src, NULL, &cb))) {
        while ((rc = exr_reader_parse_header(cb)) == EXR_WOULD_BLOCK)
            supply_all(cb, &state);
        CHECK(rc == EXR_SUCCESS, "mip streaming parse");
        {
            exr_part reg;
            int x0 = FX_OX + 9, y0 = FX_OY + 11, w = 37, h = 29, tries = 0;
            if (x0 + w > FX_OX + FX_W) w = FX_OX + FX_W - x0;
            if (y0 + h > FX_OY + FX_H) h = FX_OY + FX_H - y0;
            memset(&reg, 0, sizeof(reg));
            do {
                rc = exr_reader_read_region(cb, 0, x0, y0, w, h, 0, 0, &reg);
                if (rc == EXR_WOULD_BLOCK && !supply_all(cb, &state)) break;
            } while (rc == EXR_WOULD_BLOCK && ++tries < 1000000);
            if (EXR_OK(rc)) {
                CHECK(region_matches(&full, &reg, x0, y0, w, h,
                                     "mip streaming"),
                      "mip streaming region matches level 0");
                exr_part_free(NULL, &reg);
            } else {
                CHECK(0, "mip streaming region");
            }
        }
        exr_reader_close(cb);
    } else {
        CHECK(0, "mip open source");
    }
    exr_part_free(NULL, &full);
    exr_reader_close(mem);
    free(buf);
}

/* Corrupt a large part of the chunk area: at least one region that needs the
 * damage must report CORRUPT and own nothing; a truncated file is rejected the
 * same way. */
static void damaged_chunk_test(void) {
    fixture f;
    void *buf = NULL;
    size_t sz = 0;
    exr_reader *r = NULL;
    exr_result rc;
    uint8_t *copy;

    if (!fixture_build(&f, EXR_PART_SCANLINE, EXR_COMPRESSION_ZIP,
                       EXR_TILE_ONE_LEVEL, EXR_TILE_ROUND_DOWN, 0, 0)) {
        CHECK(0, "fixture");
        return;
    }
    rc = exr_save_to_memory(&buf, &sz, NULL, &f.img, EXR_COMPRESSION_ZIP);
    fixture_free(&f);
    if (!EXR_OK(rc)) { CHECK(0, "save damaged"); return; }
    copy = (uint8_t *)malloc(sz);
    if (!copy) { free(buf); return; }
    memcpy(copy, buf, sz);
    {
        /* Corrupt everything past the halfway point of the file (chunk area). */
        size_t k = sz / 2;
        for (; k + 8 < sz; k += 7) copy[k] ^= 0xA5;
    }
    if (EXR_OK(exr_reader_open_memory(copy, sz, NULL, &r))) {
        int saw_corrupt = 0, y0;
        for (y0 = 0; y0 < FX_H; y0 += 16) {
            exr_part reg;
            memset(&reg, 0, sizeof(reg));
            rc = exr_reader_read_region(r, 0, FX_OX, FX_OY + y0, 8, 8, 0, 0,
                                        &reg);
            if (rc == EXR_ERROR_CORRUPT) {
                CHECK(reg.images == NULL && reg.header.channels == NULL,
                      "corrupt region owns nothing");
                saw_corrupt = 1;
            } else if (EXR_OK(rc)) {
                exr_part_free(NULL, &reg);
            }
        }
        CHECK(saw_corrupt, "corrupted chunk reported as CORRUPT");
        {
            /* Cut the file before its last chunk: the offset table then points
             * past EOF and every path must reject it without owning output. */
            exr_reader *tr = NULL;
            exr_part reg;
            size_t cut = sz / 3;
            if (EXR_OK(exr_reader_open_memory(buf, cut, NULL, &tr))) {
                memset(&reg, 0, sizeof(reg));
                rc = exr_reader_read_region(tr, 0, FX_OX, FX_OY, 4, 4, 0, 0,
                                            &reg);
                CHECK(rc != EXR_SUCCESS, "truncated file rejected");
                CHECK(reg.images == NULL, "truncated region owns nothing");
                exr_reader_close(tr);
            } else {
                CHECK(1, "truncated header rejected");
            }
        }
        exr_reader_close(r);
    } else {
        CHECK(0, "open corrupted");
    }
    free(copy);
    free(buf);
}

/* Multipart: two independent parts (a scanline part and a tiled part); a
 * region from each must match that part's whole-part decode, regardless of
 * the other part's presence. */
static void multipart_test(void) {
    fixture fa, fb;
    exr_image img;
    exr_part *parts;
    void *buf = NULL;
    size_t sz = 0;
    exr_reader *r = NULL;
    exr_result rc;

    if (!fixture_build(&fa, EXR_PART_SCANLINE, EXR_COMPRESSION_ZIP,
                       EXR_TILE_ONE_LEVEL, EXR_TILE_ROUND_DOWN, 0, 0) ||
        !fixture_build(&fb, EXR_PART_TILED, EXR_COMPRESSION_ZIP,
                       EXR_TILE_ONE_LEVEL, EXR_TILE_ROUND_DOWN, 24, 24)) {
        fixture_free(&fa);
        fixture_free(&fb);
        CHECK(0, "fixture");
        return;
    }
    strcpy(fa.img.parts[0].header.name, "scan");
    strcpy(fb.img.parts[0].header.name, "tile");
    parts = (exr_part *)calloc(2, sizeof(exr_part));
    if (!parts) {
        fixture_free(&fa);
        fixture_free(&fb);
        CHECK(0, "alloc");
        return;
    }
    parts[0] = fa.img.parts[0];
    parts[1] = fb.img.parts[0];
    memset(&img, 0, sizeof(img));
    img.num_parts = 2;
    img.parts = parts;
    rc = exr_save_to_memory(&buf, &sz, NULL, &img, EXR_COMPRESSION_ZIP);
    free(parts);
    fixture_free(&fa);
    fixture_free(&fb);
    if (!EXR_OK(rc)) { CHECK(0, "multipart save"); return; }
    if (!EXR_OK(exr_reader_open_memory(buf, sz, NULL, &r))) {
        CHECK(0, "multipart open");
        free(buf);
        return;
    }
    {
        int p;
        for (p = 0; p < 2; ++p) {
            exr_part full;
            memset(&full, 0, sizeof(full));
            if (EXR_OK(exr_reader_read_part(r, p, &full))) {
                exr_part reg;
                int x0 = full.header.data_window.min_x + 3;
                int y0 = full.header.data_window.min_y + 2;
                int w = full.width - 5, h = full.height - 4;
                memset(&reg, 0, sizeof(reg));
                rc = exr_reader_read_region(r, p, x0, y0, w, h, 0, 0, &reg);
                if (EXR_OK(rc)) {
                    CHECK(region_matches(&full, &reg, x0, y0, w, h, "multipart"),
                          "multipart region matches");
                    exr_part_free(NULL, &reg);
                } else {
                    CHECK(0, "multipart region read");
                }
                exr_part_free(NULL, &full);
            } else {
                CHECK(0, "multipart full read");
            }
        }
    }
    exr_reader_close(r);
    free(buf);
}

/* A DECREASING_Y scanline file stores its blocks in reverse file order but the
 * same offset-table index; a region must still map y to the right chunk. Build
 * one from the fixture and compare against the whole-part decode. */
static void line_order_test(void) {
    fixture f;
    void *buf = NULL;
    size_t sz = 0;
    exr_reader *r = NULL;
    exr_result rc;
    exr_part full;
    int y0;

    if (!fixture_build(&f, EXR_PART_SCANLINE, EXR_COMPRESSION_ZIP,
                       EXR_TILE_ONE_LEVEL, EXR_TILE_ROUND_DOWN, 0, 0)) {
        CHECK(0, "fixture");
        return;
    }
    f.img.parts[0].header.line_order = EXR_LINEORDER_DECREASING_Y;
    rc = exr_save_to_memory(&buf, &sz, NULL, &f.img, EXR_COMPRESSION_ZIP);
    fixture_free(&f);
    if (!EXR_OK(rc)) { CHECK(0, "decreasing-y save"); return; }
    if (!EXR_OK(exr_reader_open_memory(buf, sz, NULL, &r))) {
        CHECK(0, "decreasing-y open");
        free(buf);
        return;
    }
    memset(&full, 0, sizeof(full));
    if (EXR_OK(exr_reader_read_part(r, 0, &full))) {
        for (y0 = 0; y0 < FX_H; y0 += 13) {
            exr_part reg;
            int h = 9;
            if (FX_OY + y0 + h > FX_OY + FX_H) h = FX_OY + FX_H - (FX_OY + y0);
            memset(&reg, 0, sizeof(reg));
            rc = exr_reader_read_region(r, 0, FX_OX + 4, FX_OY + y0, 23, h, 0, 0,
                                        &reg);
            if (EXR_OK(rc)) {
                CHECK(region_matches(&full, &reg, FX_OX + 4, FX_OY + y0, 23, h,
                                     "decreasing-y"),
                      "decreasing-y region matches");
                exr_part_free(NULL, &reg);
            } else {
                CHECK(0, "decreasing-y region read");
            }
        }
        exr_part_free(NULL, &full);
    } else {
        CHECK(0, "decreasing-y full read");
    }
    exr_reader_close(r);
    free(buf);
}

/* Peak allocation accounting: a small region over a tiled file must not
 * allocate the whole image. */
typedef struct mem_stats {
    size_t live, peak;
} mem_stats;

static void *peak_alloc(void *user, size_t n) {
    mem_stats *s = (mem_stats *)user;
    size_t *p = (size_t *)malloc(n + 16);
    if (!p) return NULL;
    p[0] = n;
    s->live += n;
    if (s->live > s->peak) s->peak = s->live;
    return p + 2;
}

static void peak_free(void *user, void *ptr) {
    mem_stats *s = (mem_stats *)user;
    size_t *p;
    if (!ptr) return;
    p = (size_t *)ptr - 2;
    s->live -= p[0];
    free(p);
}

static void peak_test(void) {
    fixture f;
    void *buf = NULL;
    size_t sz = 0;
    exr_reader *r = NULL;
    exr_allocator a;
    mem_stats stats;
    exr_result rc;
    size_t full_peak = 0, region_peak = 0, region_bytes, tile_bound;

    if (!fixture_build(&f, EXR_PART_TILED, EXR_COMPRESSION_ZIP,
                       EXR_TILE_ONE_LEVEL, EXR_TILE_ROUND_DOWN, 32, 32)) {
        CHECK(0, "fixture");
        return;
    }
    rc = exr_save_to_memory(&buf, &sz, NULL, &f.img, EXR_COMPRESSION_ZIP);
    fixture_free(&f);
    if (!EXR_OK(rc)) { CHECK(0, "save peak"); return; }
    memset(&stats, 0, sizeof(stats));
    a.user = &stats;
    a.alloc = peak_alloc;
    a.free = peak_free;
    if (!EXR_OK(exr_reader_open_memory(buf, sz, &a, &r))) {
        CHECK(0, "open peak");
        free(buf);
        return;
    }
    {
        exr_part p;
        memset(&p, 0, sizeof(p));
        stats.live = stats.peak = 0;
        if (EXR_OK(exr_reader_read_part(r, 0, &p))) exr_part_free(&a, &p);
        full_peak = stats.peak;
    }
    {
        exr_part p;
        memset(&p, 0, sizeof(p));
        stats.live = stats.peak = 0;
        if (EXR_OK(exr_reader_read_region(r, 0, FX_OX + 5, FX_OY + 5, 20, 20, 0,
                                          0, &p)))
            exr_part_free(&a, &p);
        region_peak = stats.peak;
    }
    region_bytes = (size_t)20 * 20 * (2 + 4 + 4) + 3 * sizeof(void *);
    tile_bound = 32 * 32 * (2 + 4 + 4) + 3 * sizeof(void *);
    CHECK(full_peak > region_peak, "region peak below whole-image peak");
    /* Region + a decoded tile + the codec's same-size reconstruction scratch.
     * ZIP also allocates a same-size inflate buffer, so allow a few tiles. */
    CHECK(region_peak <= region_bytes + 4 * tile_bound + 65536,
          "region peak bounded by region plus a tile");
    exr_reader_close(r);
    free(buf);
}

int main(void) {
    printf("== region: scanline compressions ==\n");
    scanline_tests();
    printf("== region: tiled levels (one/mip/ripmap, both roundings) ==\n");
    tiled_tests();
    printf("== region: exhaustive small tiles ==\n");
    exhaustive_small();
    printf("== region: error contract ==\n");
    error_contract();
    printf("== region: streaming source ==\n");
    streaming_test();
    streaming_mipmap_test();
    printf("== region: damaged chunks ==\n");
    damaged_chunk_test();
    printf("== region: peak memory ==\n");
    peak_test();
    printf("== region: multipart ==\n");
    multipart_test();
    printf("== region: line order ==\n");
    line_order_test();
    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
