/*
    This file is part of darktable,
    copyright (c) 2019 Aurélien Pierre.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * The image doctor surgically reconstruct damaged, missing or inconsistent parts
 * of your image using the valid data from other channels.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "bauhaus/bauhaus.h"
#include "common/darktable.h"
#include "common/fast_guided_filter.h"
#include "common/opencl.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "gui/presets.h"
#include "iop/iop_api.h"
#include "common/iop_group.h"

DT_MODULE_INTROSPECTION(1, dt_iop_doctor_params_t)

/** Note :
 * we use finite-math-only and fast-math because divisions by zero are manually avoided in the code
 * fp-contract=fast enables hardware-accelerated Fused Multiply-Add
 * the rest is loop reorganization and vectorization optimization
 **/
#if defined(__GNUC__)
#pragma GCC optimize ("unroll-loops", "tree-loop-if-convert", \
                      "tree-loop-distribution", "no-strict-aliasing", \
                      "loop-interchange", "loop-nest-optimize", "tree-loop-im", \
                      "unswitch-loops", "tree-loop-ivcanon", "ira-loop-pressure", \
                      "split-ivs-in-unroller", "variable-expansion-in-unroller", \
                      "split-loops", "ivopts", "predictive-commoning",\
                      "tree-loop-linear", "loop-block", "loop-strip-mine", \
                      "finite-math-only", "fp-contract=fast", "fast-math", \
                      "tree-vectorize")
#endif

typedef struct dt_iop_doctor_params_t
{
  int scales, iterations;
  float luma_strength, luma_feathering;
  float chroma_strength, chroma_feathering;
  float fringes_strength, fringes_feathering;
  float sharpness_strength, sharpness_feathering;
  float highlight_clipping, lowlight_clipping;
} dt_iop_doctor_params_t;

typedef struct dt_iop_doctor_gui_data_t
{
  GtkNotebook *notebook;
  GtkWidget *scales, *iterations;
  GtkWidget *luma_feathering, *luma_strength;
  GtkWidget *chroma_feathering, *chroma_strength;
  GtkWidget *fringes_feathering, *fringes_strength;
  GtkWidget *highlight_clipping, *lowlight_clipping;
  GtkWidget *sharpness_feathering, *sharpness_strength;
} dt_iop_doctor_gui_data_t;

typedef dt_iop_doctor_params_t dt_iop_doctor_data_t;

const char *name()
{
  return _("image doctor");
}

int default_group()
{
  return IOP_GROUP_CORRECT;
}

int flags()
{
  return IOP_FLAGS_SUPPORTS_BLENDING;
}

int default_colorspace(dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  return iop_cs_rgb;
}

static inline float sqf(const float x)
{
  return x * x;
}

static inline float clamp(const float x)
{
  return fmaxf(fminf(x, 1.0f), -1.0f);
}

static inline void denoise_chroma(float *const image[3], float *const residual_out[3],
                                  const size_t width, const size_t height,
                                  const int radius, const float scale,
                                  const float chroma_feathering, const float chroma_strength)
{
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(image, residual_out, height, width) \
schedule(static) collapse(2) aligned(residual_out, image:64)
#endif
  for(size_t i = 0; i < height; ++i)
    for(size_t j = 0; j < width; ++j)
    {
      const size_t index = i * width + j;
      residual_out[0][index] = image[1][index] - image[0][index]; // G - R
      residual_out[1][index] = image[0][index] - image[1][index]; // R - G
      residual_out[2][index] = image[1][index] - image[2][index]; // G - B
      residual_out[3][index] = image[2][index] - image[1][index]; // B - G
    }

  for(int c = 0; c < 4; ++c)
    fast_guided_filter_rgb(residual_out[c], residual_out[c], residual_out[c], width, height, 1, radius, chroma_feathering, scale, FALSE);

  const float b = chroma_strength;
  const float a = 1.0f - b;

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(image, residual_out, height, width, a, b) \
schedule(static) collapse(2) aligned(residual_out, image:64)
#endif
  for(size_t i = 0; i < height; ++i)
    for(size_t j = 0; j < width; ++j)
    {
      const size_t index = i * width + j;
      const float R = image[0][index];
      const float G = image[1][index];
      const float B = image[2][index];
      image[0][index] = a * R + b * (G - clamp(residual_out[0][index]));
      image[1][index] = a * G + b * (R - clamp(residual_out[1][index]) + B - clamp(residual_out[3][index])) / 2.0f;
      image[2][index] = a * B + b * (G - clamp(residual_out[2][index]));
    }
}


static inline void luma_mask(float *const input[3], float *const restrict luma,
                             const size_t width, const size_t height)
{
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(input, luma, height, width) \
schedule(static) collapse(2) aligned(luma, input:64)
#endif
  for(size_t i = 0; i < height; ++i)
    for(size_t j = 0; j < width; ++j)
    {
      const size_t index = i * width + j;
      luma[index] = sqrtf(sqf(input[0][index]) + sqf(input[1][index]) + sqf(input[2][index]));
    }
}

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece,
             const void *const restrict ivoid, void *const restrict ovoid,
             const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  const dt_iop_doctor_data_t *const d = (const dt_iop_doctor_data_t *const)piece->data;

  const float *const restrict in = dt_check_sse_aligned((float *const)ivoid);
  float *const restrict out = dt_check_sse_aligned((float *const)ovoid);

  const size_t width = roi_in->width;
  const size_t height = roi_in->height;
  const size_t ch = 4;

  float *const residual_out[4] = { dt_alloc_sse_ps(width * height),
                                   dt_alloc_sse_ps(width * height),
                                   dt_alloc_sse_ps(width * height),
                                   dt_alloc_sse_ps(width * height)};

  float *const RGB_high_freq[4] = { dt_alloc_sse_ps(width * height),
                                          dt_alloc_sse_ps(width * height),
                                          dt_alloc_sse_ps(width * height),
                                          dt_alloc_sse_ps(width * height)};

  float *const out_RGB[3] = { dt_alloc_sse_ps(width * height),
                              dt_alloc_sse_ps(width * height),
                              dt_alloc_sse_ps(width * height) };

  float *const in_RGB[3] = { dt_alloc_sse_ps(width * height),
                              dt_alloc_sse_ps(width * height),
                              dt_alloc_sse_ps(width * height) };

  float *const restrict luma = dt_alloc_sse_ps(width * height);
  float *const restrict luma_low_freq = dt_alloc_sse_ps(width * height);
  float *const restrict luma_high_freq = dt_alloc_sse_ps(width * height);

  const float epsilon = d->iterations * 100.0f;
  const float luma_feathering = 1.f / powf(10.f, 2.0f * d->luma_feathering);
  const float luma_strength = 1.0f - (d->luma_strength /  epsilon);

  const float chroma_feathering = 1.f / powf(10.f, 2.0f * d->chroma_feathering);
  const float chroma_strength = d->chroma_strength / epsilon;

  const float fringes_feathering = 1.f / powf(10.f, 2.0f * d->fringes_feathering);
  const float fringes_strength = d->fringes_strength / epsilon;

  const float sharpness_feathering = 1.f / powf(10.f, 2.0f * d->sharpness_feathering);
  const float sharpness_strength = d->sharpness_strength / epsilon;

  //const float highlight_clipping = d->highlight_clipping / 100.0f;
  //const float lowlight_clipping = exp2f(d->lowlight_clipping);

  const int run_chroma = (chroma_strength != 0.0f);
  const int run_luma = (luma_strength != 1.0f);
  const int run_fringes = (fringes_strength != 0.0f);
  const int run_sharpness = (sharpness_strength != 0.0f);

  // Peel the array of structs into RGB layers (struct of arrays)
#ifdef _OPENMP
#pragma omp parallel for default(none) \
dt_omp_firstprivate(out_RGB, in_RGB, in, height, width, ch) \
schedule(static) collapse(3)
#endif
  for(size_t i = 0; i < height; ++i)
    for(size_t j = 0; j < width; ++j)
      for(size_t c = 0; c < 3; ++c)
      {
        out_RGB[c][(i * width + j)] = in_RGB[c][(i * width + j)] = in[(i * width + j) * ch + c];
      }

  for(int k = 0; k < d->scales; k++)
  {
    const int radius = MAX(1, (d->scales - k) / piece->iscale * roi_in->scale);
    float scale;
    if(radius % 4 == 0)
      scale = 4.0f;
    else if(radius % 3 == 0)
      scale = 3.0f;
    else if(radius % 2 == 0)
      scale = 2.0f;
    else
      scale = 1.f;

    for(int outer_iter = 0; outer_iter < d->iterations; outer_iter++)
    {
      // Chroma denoise
      if(run_chroma)
        denoise_chroma(out_RGB, residual_out, width, height, scale, radius, chroma_feathering, chroma_strength);

      // Re-apply the luminance high frequency on top of RGB channels
      if(run_luma)
      {
        // Get the RGB channel low frequency
        // TODO: use a true RGB guided filter ?
        for(int c = 0; c < 3; c++)
          fast_guided_filter_rgb(in_RGB[c], out_RGB[c], residual_out[c], width, height, 1, radius, luma_feathering, scale, TRUE);

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(out_RGB, luma_low_freq, luma_high_freq, luma, residual_out, height, width, ch, luma_strength) \
schedule(static) collapse(3) aligned(RGB_high_freq, luma_high_freq, out_RGB:64)
#endif
        for(size_t i = 0; i < height; i++)
        {
          for(size_t j = 0; j < width; j++)
          {
            for(size_t c = 0; c < 3; c++)
            {
              const size_t index = (i * width + j);
              const float LF = residual_out[c][index];
              const float HF = out_RGB[c][index] - residual_out[c][index];
              out_RGB[c][index] = luma_strength * HF + LF;
            }
          }
        }
      }

      if(run_fringes)
      {
        // Defringe
        // note : RGB_high_freq actually contains the low freq here, we just reuse the buffer

        // guide R with G
        fast_guided_filter_rgb(out_RGB[0], out_RGB[1], RGB_high_freq[0], width, height, 1, radius, fringes_feathering, scale, TRUE);

        // guide B with G
        fast_guided_filter_rgb(out_RGB[2], out_RGB[1], RGB_high_freq[2], width, height, 1, radius, fringes_feathering, scale, TRUE);

        // guide G with B + R
#ifdef _OPENMP
#pragma omp parallel for default(none) \
dt_omp_firstprivate(out_RGB, RGB_high_freq, residual_out, luma, height, width, ch, fringes_strength) \
schedule(static) collapse(2)
#endif
        for(size_t i = 0; i < height; ++i)
        {
          for(size_t j = 0; j < width; ++j)
          {
            const size_t index = i * width + j;
            residual_out[0][index] = (out_RGB[0][index] + out_RGB[2][index]) / 2.0f;
          }
        }
        fast_guided_filter_rgb(out_RGB[1], residual_out[0], RGB_high_freq[1], width, height, 1, radius, fringes_feathering, scale, TRUE);

        // Guide RGB by themselves
        for(int c = 0; c < 3; c++)
          fast_guided_filter_rgb(out_RGB[c], out_RGB[c], residual_out[c], width, height, 1, radius, fringes_feathering, scale, TRUE);


#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(out_RGB, RGB_high_freq, residual_out, luma, height, width, ch, fringes_strength) \
schedule(static) collapse(3) aligned(out_RGB, residual_out, RGB_high_freq:64)
#endif
        for(size_t i = 0; i < height; ++i)
        {
          for(size_t j = 0; j < width; ++j)
          {
            for(size_t c = 0; c < 3; ++c)
            {
              const size_t index = i * width + j;
              const float high_freq_diff = residual_out[c][index] - RGB_high_freq[c][index];
              out_RGB[c][index] = out_RGB[c][index] + fringes_strength * high_freq_diff;
            }
          }
        }
      }

      if(run_sharpness)
      {
        luma_mask(out_RGB, luma, width, height);
        for(int c = 0; c < 3; c++)
          fast_guided_filter_rgb(out_RGB[c], luma, residual_out[c], width, height, 1, radius, sharpness_feathering, scale, FALSE);

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(out_RGB, RGB_high_freq, residual_out, luma, height, width, ch, sharpness_strength) \
schedule(static) collapse(3) aligned(out_RGB, residual_out:64)
#endif
        for(size_t i = 0; i < height; ++i)
        {
          for(size_t j = 0; j < width; ++j)
          {
            for(size_t c = 0; c < 3; ++c)
            {
              const size_t index = i * width + j;
              const float high_freq = out_RGB[c][index] - residual_out[c][index];
              out_RGB[c][index] += sharpness_strength * high_freq;
            }
          }
        }

      // TODO: code inpanting
      // https://www.researchgate.net/profile/Shuozhong_Wang/publication/220663968_Simultaneous_inpainting_for_image_structure_and_texture_using_anisotropic_heat_transfer_model/links/53d8eab70cf2a19eee839242/Simultaneous-inpainting-for-image-structure-and-texture-using-anisotropic-heat-transfer-model.pdf

      }
    }
  }

  // Repack the RGB layers to an array of stucts
#ifdef _OPENMP
#pragma omp parallel for default(none) \
dt_omp_firstprivate(out_RGB, out, height, width, ch) \
schedule(static) collapse(3)
#endif
  for(size_t i = 0; i < height; ++i)
    for(size_t j = 0; j < width; ++j)
      for(size_t c = 0; c < 3; ++c)
      {
        out[(i * width + j) * ch + c] = out_RGB[c][i * width + j];
      }

  for(int c = 0; c < 3; c++)
  {
    dt_free_align(out_RGB[c]);
    dt_free_align(in_RGB[c]);

  }
  for(int c = 0; c < 4; c++)
  {
    dt_free_align(residual_out[c]);
    dt_free_align(RGB_high_freq[c]);
  }
  dt_free_align(luma_low_freq);
  dt_free_align(luma_high_freq);
  dt_free_align(luma);
}

void init(dt_iop_module_t *module)
{
  module->params = (dt_iop_params_t *)malloc(sizeof(dt_iop_doctor_params_t));
  module->default_params = (dt_iop_params_t *)malloc(sizeof(dt_iop_doctor_params_t));
  module->default_enabled = 0;
  module->params_size = sizeof(dt_iop_doctor_params_t);
  module->gui_data = NULL;
  dt_iop_doctor_params_t tmp = (dt_iop_doctor_params_t){ .scales = 3,
                                                         .iterations = 3,
                                                         .luma_strength = 15.f,
                                                         .luma_feathering = 2.5f,
                                                         .chroma_strength = 50.f,
                                                         .chroma_feathering = 3.0f, // chroma
                                                         .fringes_strength = 5.f,
                                                         .fringes_feathering = 2.0f, // fringes
                                                         .sharpness_strength = 20.f,
                                                         .sharpness_feathering = 1.5f,   // sharpness
                                                         .highlight_clipping = 99.0f,
                                                         .lowlight_clipping = -12.0f}; // clipping
  memcpy(module->params, &tmp, sizeof(dt_iop_doctor_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_doctor_params_t));
}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
}

void init_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = malloc(sizeof(dt_iop_doctor_data_t));
  self->commit_params(self, self->default_params, pipe, piece);
}

void cleanup_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_update(struct dt_iop_module_t *self)
{
  dt_iop_module_t *module = (dt_iop_module_t *)self;
  dt_iop_doctor_gui_data_t *g = (dt_iop_doctor_gui_data_t *)self->gui_data;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)module->params;

  dt_bauhaus_slider_set_soft(g->scales, p->scales);
  dt_bauhaus_slider_set_soft(g->iterations, p->iterations);

  dt_bauhaus_slider_set_soft(g->luma_feathering, p->luma_feathering);
  dt_bauhaus_slider_set_soft(g->luma_strength, p->luma_strength);

  dt_bauhaus_slider_set_soft(g->chroma_feathering, p->chroma_feathering);
  dt_bauhaus_slider_set_soft(g->chroma_strength, p->chroma_strength);

  dt_bauhaus_slider_set_soft(g->fringes_feathering, p->fringes_feathering);
  dt_bauhaus_slider_set_soft(g->fringes_strength, p->fringes_strength);

  dt_bauhaus_slider_set_soft(g->highlight_clipping, p->highlight_clipping);
  dt_bauhaus_slider_set_soft(g->lowlight_clipping, p->lowlight_clipping);

  dt_bauhaus_slider_set_soft(g->sharpness_feathering, p->sharpness_feathering);
  dt_bauhaus_slider_set_soft(g->sharpness_strength, p->sharpness_strength);
}

static void scales_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->scales = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void iterations_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->iterations = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void luma_feathering_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->luma_feathering = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void luma_strength_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->luma_strength = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void chroma_feathering_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->chroma_feathering = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void chroma_strength_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->chroma_strength = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void fringes_feathering_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->fringes_feathering = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void fringes_strength_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->fringes_strength = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void sharpness_feathering_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->sharpness_feathering = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void sharpness_strength_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->sharpness_strength = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void highlight_clipping_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->highlight_clipping = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void lowlight_clipping_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->lowlight_clipping = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void gui_init(dt_iop_module_t *self)
{
  self->gui_data = (dt_iop_gui_data_t *)malloc(sizeof(dt_iop_doctor_gui_data_t));
  dt_iop_doctor_gui_data_t *g = (dt_iop_doctor_gui_data_t *)self->gui_data;
  //dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;

  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);

  g->scales = dt_bauhaus_slider_new_with_range(self, 1, 6, 1, 1, 0);
  dt_bauhaus_slider_enable_soft_boundaries(g->scales, 1, 20);
  dt_bauhaus_widget_set_label(g->scales, NULL, _("filtering scales"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->scales, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->scales), "value-changed", G_CALLBACK(scales_callback), self);

  g->iterations = dt_bauhaus_slider_new_with_range(self, 1, 6, 1, 1, 0);
  dt_bauhaus_slider_enable_soft_boundaries(g->iterations, 1, 20);
  dt_bauhaus_widget_set_label(g->iterations, NULL, _("filtering iterations"));
  gtk_box_pack_start(GTK_BOX(self->widget), g->iterations, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->iterations), "value-changed", G_CALLBACK(iterations_callback), self);


  g->notebook = GTK_NOTEBOOK(gtk_notebook_new());
  GtkWidget *page1 = GTK_WIDGET(gtk_box_new(GTK_ORIENTATION_VERTICAL, 0));
  GtkWidget *page2 = GTK_WIDGET(gtk_box_new(GTK_ORIENTATION_VERTICAL, 0));
  GtkWidget *page3 = GTK_WIDGET(gtk_box_new(GTK_ORIENTATION_VERTICAL, 0));
  GtkWidget *page4 = GTK_WIDGET(gtk_box_new(GTK_ORIENTATION_VERTICAL, 0));
  GtkWidget *page5 = GTK_WIDGET(gtk_box_new(GTK_ORIENTATION_VERTICAL, 0));

  gtk_notebook_append_page(GTK_NOTEBOOK(g->notebook), page1, gtk_label_new(_("luma")));
  gtk_notebook_append_page(GTK_NOTEBOOK(g->notebook), page2, gtk_label_new(_("chroma")));
  gtk_notebook_append_page(GTK_NOTEBOOK(g->notebook), page3, gtk_label_new(_("fringes")));
  gtk_notebook_append_page(GTK_NOTEBOOK(g->notebook), page4, gtk_label_new(_("clipped")));
  gtk_notebook_append_page(GTK_NOTEBOOK(g->notebook), page5, gtk_label_new(_("sharpness")));
  gtk_widget_show_all(GTK_WIDGET(gtk_notebook_get_nth_page(g->notebook, 0)));
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->notebook), FALSE, FALSE, 0);

  gtk_container_child_set(GTK_CONTAINER(g->notebook), page1, "tab-expand", TRUE, "tab-fill", TRUE, NULL);
  gtk_container_child_set(GTK_CONTAINER(g->notebook), page2, "tab-expand", TRUE, "tab-fill", TRUE, NULL);
  gtk_container_child_set(GTK_CONTAINER(g->notebook), page3, "tab-expand", TRUE, "tab-fill", TRUE, NULL);
  gtk_container_child_set(GTK_CONTAINER(g->notebook), page4, "tab-expand", TRUE, "tab-fill", TRUE, NULL);
  gtk_container_child_set(GTK_CONTAINER(g->notebook), page5, "tab-expand", TRUE, "tab-fill", TRUE, NULL);

  g->luma_feathering = dt_bauhaus_slider_new_with_range(self, 0.1, 4., 0.2, 5., 2);
  dt_bauhaus_slider_set_format(g->luma_feathering, "%.2f dB");
  dt_bauhaus_widget_set_label(g->luma_feathering, NULL, _("edges sensitivity"));
  gtk_box_pack_start(GTK_BOX(page1), g->luma_feathering, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->luma_feathering), "value-changed", G_CALLBACK(luma_feathering_callback), self);

  g->luma_strength = dt_bauhaus_slider_new_with_range(self, 0., 100., 0.2, 100., 2);
  dt_bauhaus_slider_set_format(g->luma_strength, "%.2f %%");
  dt_bauhaus_widget_set_label(g->luma_strength , NULL, _("denoising"));
  gtk_box_pack_start(GTK_BOX(page1), g->luma_strength , FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->luma_strength ), "value-changed", G_CALLBACK(luma_strength_callback), self);

  g->chroma_feathering = dt_bauhaus_slider_new_with_range(self, 0.1, 4., 0.2, 5., 2);
  dt_bauhaus_slider_set_format(g->chroma_feathering, "%.2f dB");
  dt_bauhaus_widget_set_label(g->chroma_feathering, NULL, _("edges sensitivity"));
  gtk_box_pack_start(GTK_BOX(page2), g->chroma_feathering, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->chroma_feathering), "value-changed", G_CALLBACK(chroma_feathering_callback), self);

  g->chroma_strength = dt_bauhaus_slider_new_with_range(self, 0., 100., 0.2, 100., 2);
  dt_bauhaus_slider_set_format(g->chroma_strength, "%.2f %%");
  dt_bauhaus_widget_set_label(g->chroma_strength , NULL, _("denoising"));
  gtk_box_pack_start(GTK_BOX(page2), g->chroma_strength, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->chroma_strength), "value-changed", G_CALLBACK(chroma_strength_callback), self);

  g->fringes_feathering = dt_bauhaus_slider_new_with_range(self, 0.1, 4., 0.2, 5., 2);
  dt_bauhaus_slider_set_format(g->fringes_feathering, "%.2f dB");
  dt_bauhaus_widget_set_label(g->fringes_feathering, NULL, _("edges sensitivity"));
  gtk_box_pack_start(GTK_BOX(page3), g->fringes_feathering, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->fringes_feathering), "value-changed", G_CALLBACK(fringes_feathering_callback), self);

  g->fringes_strength = dt_bauhaus_slider_new_with_range(self, 0., 100., 0.2, 100., 2);
  dt_bauhaus_slider_set_format(g->fringes_strength, "%.2f %%");
  dt_bauhaus_widget_set_label(g->fringes_strength , NULL, _("defringing"));
  gtk_box_pack_start(GTK_BOX(page3), g->fringes_strength, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->fringes_strength), "value-changed", G_CALLBACK(fringes_strength_callback), self);

  g->highlight_clipping = dt_bauhaus_slider_new_with_range(self, 0., 120., 1., 100., 2);
  dt_bauhaus_slider_set_format(g->highlight_clipping, "%.2f %%");
  dt_bauhaus_widget_set_label(g->highlight_clipping, NULL, _("highlights clipping"));
  gtk_box_pack_start(GTK_BOX(page4), g->highlight_clipping, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->highlight_clipping), "value-changed", G_CALLBACK(highlight_clipping_callback), self);

  g->lowlight_clipping = dt_bauhaus_slider_new_with_range(self, -16., 0., 1., -12., 2);
  dt_bauhaus_slider_set_format(g->lowlight_clipping, "%.2f EV");
  dt_bauhaus_widget_set_label(g->lowlight_clipping, NULL, _("lowlights clipping"));
  gtk_box_pack_start(GTK_BOX(page4), g->lowlight_clipping, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->lowlight_clipping), "value-changed", G_CALLBACK(lowlight_clipping_callback), self);

  g->sharpness_feathering = dt_bauhaus_slider_new_with_range(self, 0.1, 4., 0.2, 5., 2);
  dt_bauhaus_slider_set_format(g->sharpness_feathering, "%.2f dB");
  dt_bauhaus_widget_set_label(g->sharpness_feathering, NULL, _("edges sensitivity"));
  gtk_box_pack_start(GTK_BOX(page5), g->sharpness_feathering, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->sharpness_feathering), "value-changed", G_CALLBACK(sharpness_feathering_callback), self);

  g->sharpness_strength = dt_bauhaus_slider_new_with_range(self, 0., 100., 1., 0., 2);
  dt_bauhaus_slider_set_format(g->sharpness_strength , "%.2f %%");
  dt_bauhaus_widget_set_label(g->sharpness_strength , NULL, _("sharpening"));
  gtk_box_pack_start(GTK_BOX(page5), g->sharpness_strength , FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->sharpness_strength ), "value-changed", G_CALLBACK(sharpness_strength_callback), self);
}

void gui_cleanup(struct dt_iop_module_t *self)
{
  free(self->gui_data);
  self->gui_data = NULL;
}
