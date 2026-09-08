/* Benchmark-only public API adapters. All allocation and encoder lifetime
 * cleanup stays on the C side; no internal oracle symbols or global tracing. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <aom/aom_encoder.h>
#include <aom/aom_decoder.h>
#include <aom/aomcx.h>
#include <aom/aomdx.h>
#include <EbSvtAv1Enc.h>

typedef struct { uint8_t *data; size_t len; } Output;
void zm_av1_free(uint8_t *p) { free(p); }
static int append(Output *o, const uint8_t *p, size_t n) {
    if (n > SIZE_MAX - o->len) return -1;
    if (!n) return 0;
    uint8_t *next = realloc(o->data, o->len + n);
    if (!next) return -1;
    memcpy(next + o->len, p, n);
    o->data = next;
    o->len += n;
    return 0;
}

/* First comparison envelope: packed 8-bit I420, explicitly configured
 * quantizer and effort. No claim that equal quantizers imply equal quality. */
int zm_libaom(const uint8_t *pixels, unsigned w, unsigned h,
              unsigned q, unsigned speed, unsigned threads, Output *out) {
    aom_codec_ctx_t ctx;
    aom_codec_enc_cfg_t cfg;
    aom_image_t image;
    int initialized = 0, rc = -1;
    memset(&ctx, 0, sizeof(ctx));
    if (aom_codec_enc_config_default(aom_codec_av1_cx(), &cfg, AOM_USAGE_ALL_INTRA)) return -2;
    cfg.g_w = w; cfg.g_h = h; cfg.g_threads = threads;
    cfg.g_timebase.num = 1; cfg.g_timebase.den = 1;
    cfg.g_limit = 1; cfg.g_lag_in_frames = 0;
    cfg.rc_end_usage = AOM_Q;
    if (aom_codec_enc_init(&ctx, aom_codec_av1_cx(), &cfg, 0)) return -3;
    initialized = 1;
#define CTRL(id, val) do { if (aom_codec_control(&ctx, id, val)) { rc = -4; goto done; } } while (0)
    CTRL(AOME_SET_CPUUSED, (int)speed);
    CTRL(AOME_SET_CQ_LEVEL, q);
    /* An explicit common SB64 arm; record it in the protocol. */
    CTRL(AV1E_SET_SUPERBLOCK_SIZE, AOM_SUPERBLOCK_SIZE_64X64);
    if (!aom_img_wrap(&image, AOM_IMG_FMT_I420, w, h, 1, (uint8_t *)pixels)) { rc = -5; goto done; }
    if (aom_codec_encode(&ctx, &image, 0, 1, AOM_EFLAG_FORCE_KF)) { rc = -6; goto done; }
    for (unsigned flush = 0; flush < 2; ++flush) {
        aom_codec_iter_t it = NULL;
        const aom_codec_cx_pkt_t *pkt;
        while ((pkt = aom_codec_get_cx_data(&ctx, &it))) {
            if (pkt->kind == AOM_CODEC_CX_FRAME_PKT &&
                append(out, pkt->data.frame.buf, pkt->data.frame.sz)) { rc = -7; goto done; }
        }
        if (!flush && aom_codec_encode(&ctx, NULL, 1, 1, 0)) { rc = -8; goto done; }
    }
    rc = out->len ? 0 : -9;
done:
    if (initialized) aom_codec_destroy(&ctx);
    return rc;
#undef CTRL
}

int zm_c_svt(const uint8_t *pixels, unsigned w, unsigned h,
             unsigned q, unsigned speed, unsigned threads, Output *out) {
    EbComponentType *ctx = NULL;
    EbSvtAv1EncConfiguration cfg;
    int initialized = 0, rc = -1;
    memset(&cfg, 0, sizeof(cfg));
    if (svt_av1_enc_init_handle(&ctx, &cfg) != EB_ErrorNone) {
        if (ctx) svt_av1_enc_deinit_handle(ctx);
        return -2;
    }
    cfg.source_width = w; cfg.source_height = h;
    cfg.enc_mode = (int8_t)speed; cfg.qp = q;
    cfg.rate_control_mode = 0; cfg.aq_mode = 0;
    cfg.avif = 1; cfg.encoder_bit_depth = 8;
    cfg.encoder_color_format = EB_YUV420;
    /* lp is SVT's parallelism level, NOT an exact OS thread count. */
    cfg.level_of_parallelism = threads;
    cfg.frame_rate_numerator = 1; cfg.frame_rate_denominator = 1;
    if (svt_av1_enc_set_parameter(ctx, &cfg) != EB_ErrorNone) { rc = -3; goto done; }
    if (svt_av1_enc_init(ctx) != EB_ErrorNone) { rc = -4; goto done; }
    initialized = 1;
    EbSvtIOFormat io;
    memset(&io, 0, sizeof(io));
    io.luma = (uint8_t *)pixels;
    io.cb = (uint8_t *)pixels + (size_t)w*h;
    io.cr = io.cb + (size_t)(w/2)*(h/2);
    io.y_stride = w; io.cb_stride = w/2; io.cr_stride = w/2;
    EbBufferHeaderType in, eos;
    memset(&in, 0, sizeof(in)); memset(&eos, 0, sizeof(eos));
    in.size = sizeof(in); in.p_buffer = (uint8_t *)&io;
    in.n_filled_len = w*h + (w*h)/2; in.pic_type = EB_AV1_INVALID_PICTURE;
    eos.size = sizeof(eos); eos.flags = EB_BUFFERFLAG_EOS;
    eos.pic_type = EB_AV1_INVALID_PICTURE;
    if (svt_av1_enc_send_picture(ctx, &in) != EB_ErrorNone ||
        svt_av1_enc_send_picture(ctx, &eos) != EB_ErrorNone) { rc = -5; goto done; }
    for (;;) {
        EbBufferHeaderType *pkt = NULL;
        EbErrorType err = svt_av1_enc_get_packet(ctx, &pkt, 1);
        if (err != EB_ErrorNone) {
            if (pkt) svt_av1_enc_release_out_buffer(&pkt);
            rc = -6; goto done;
        }
        if (!pkt) { rc = -7; goto done; }
        int last = !!(pkt->flags & EB_BUFFERFLAG_EOS);
        int copied = append(out, pkt->p_buffer, pkt->n_filled_len);
        svt_av1_enc_release_out_buffer(&pkt);
        if (copied) { rc = -8; goto done; }
        if (last) break;
    }
    rc = out->len ? 0 : -9;
done:
    if (initialized) svt_av1_enc_deinit(ctx);
    svt_av1_enc_deinit_handle(ctx);
    return rc;
}

int zm_check_decode(const uint8_t *obu, size_t len, unsigned w, unsigned h) {
    aom_codec_ctx_t ctx;
    aom_codec_dec_cfg_t cfg = {0};
    cfg.threads = 1;
    memset(&ctx, 0, sizeof(ctx));
    if (aom_codec_dec_init(&ctx, aom_codec_av1_dx(), &cfg, 0)) return -1;
    int rc = -2;
    if (!aom_codec_decode(&ctx, obu, len, NULL)) {
        aom_codec_iter_t it = NULL;
        aom_image_t *img = aom_codec_get_frame(&ctx, &it);
        if (img && img->d_w == w && img->d_h == h && img->bit_depth == 8 &&
            img->x_chroma_shift == 1 && img->y_chroma_shift == 1 &&
            !aom_codec_get_frame(&ctx, &it)) rc = 0;
    }
    aom_codec_destroy(&ctx);
    return rc;
}

/* Copy decoded I420 out before destroying the decoder. Pixel scoring never
 * borrows a decoder-owned buffer after its lifetime. */
int zm_decode_i420(const uint8_t *obu, size_t len, unsigned w, unsigned h, uint8_t *dst) {
    aom_codec_ctx_t ctx;
    aom_codec_dec_cfg_t cfg = {0}; cfg.threads = 1;
    memset(&ctx, 0, sizeof(ctx));
    if (aom_codec_dec_init(&ctx, aom_codec_av1_dx(), &cfg, 0)) return -1;
    int rc = -2;
    if (!aom_codec_decode(&ctx, obu, len, NULL)) {
        aom_codec_iter_t it = NULL;
        aom_image_t *img = aom_codec_get_frame(&ctx, &it);
        if (img && img->d_w == w && img->d_h == h && img->bit_depth == 8 &&
            img->x_chroma_shift == 1 && img->y_chroma_shift == 1 && img->range == AOM_CR_STUDIO_RANGE) {
            rc = 0;
            for (unsigned p=0; p<3; ++p) {
                unsigned pw=p?w/2:w, ph=p?h/2:h;
                for (unsigned y=0; y<ph; ++y) {
                    const uint8_t *row=img->planes[p]+(size_t)y*img->stride[p];
                    if (img->fmt & AOM_IMG_FMT_HIGHBITDEPTH) {
                        for (unsigned x=0; x<pw; ++x) dst[x]=(uint8_t)((const uint16_t*)row)[x];
                    } else memcpy(dst,row,pw);
                    dst+=pw;
                }
            }
            if (aom_codec_get_frame(&ctx, &it)) rc=-3;
        }
    }
    aom_codec_destroy(&ctx); return rc;
}
