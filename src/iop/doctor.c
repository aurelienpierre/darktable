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
  float highlight_clipping, lowlight_clipping, structure_threshold, update_speed;
  int reconstruct_iterations;
} dt_iop_doctor_params_t;

typedef struct dt_iop_doctor_gui_data_t
{
  GtkNotebook *notebook;
  GtkWidget *scales, *iterations;
  GtkWidget *luma_feathering, *luma_strength;
  GtkWidget *chroma_feathering, *chroma_strength;
  GtkWidget *fringes_feathering, *fringes_strength;
  GtkWidget *highlight_clipping, *lowlight_clipping;
  GtkWidget *sharpness_feathering, *sharpness_strength, *structure_threshold, *update_speed, *reconstruct_iterations;
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


#ifdef _OPENMP
#pragma omp declare simd
#endif
static inline float sqf(const float x)
{
  return x * x;
}

#ifdef _OPENMP
#pragma omp declare simd
#endif
static inline float cbf(const float x)
{
  return x * x * x;
}


 #ifdef _OPENMP
#pragma omp declare simd
#endif
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


static inline void denoise_chroma_crossed(float *const image[3], float *const residual_out[3],
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

  float *const tmp[4] = { dt_alloc_sse_ps(width * height),
                          dt_alloc_sse_ps(width * height),
                          dt_alloc_sse_ps(width * height),
                          dt_alloc_sse_ps(width * height)};


  fast_guided_filter_rgb(residual_out[0], residual_out[2], tmp[0], width, height, 1, radius, chroma_feathering, scale, FALSE);
  fast_guided_filter_rgb(residual_out[1], residual_out[3], tmp[1], width, height, 1, radius, chroma_feathering, scale, FALSE);
  fast_guided_filter_rgb(residual_out[2], residual_out[0], tmp[2], width, height, 1, radius, chroma_feathering, scale, FALSE);
  fast_guided_filter_rgb(residual_out[3], residual_out[1], tmp[3], width, height, 1, radius, chroma_feathering, scale, FALSE);

  const float b = chroma_strength;
  const float a = 1.0f - b;

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(image, residual_out, tmp, height, width, a, b) \
schedule(static) collapse(2) aligned(tmp, residual_out, image:64)
#endif
  for(size_t i = 0; i < height; ++i)
    for(size_t j = 0; j < width; ++j)
    {
      const size_t index = i * width + j;
      const float R = image[0][index];
      const float G = image[1][index];
      const float B = image[2][index];
      image[0][index] = a * R + b * (G - residual_out[0][index] + b * clamp(tmp[0][index]) / 2.0f);
      image[1][index] = a * G + b * (R - residual_out[1][index] + b * clamp(tmp[1][index]) / 2.0f + B - residual_out[3][index] + b * clamp(tmp[3][index]) / 2.0f) / 2.0f;
      image[2][index] = a * B + b * (G - residual_out[2][index] + b * clamp(tmp[2][index]) / 2.0f);
    }

  for(int c = 0; c < 3; c++)
    dt_free_align(tmp[c]);
}

static void heat_PDE_inpanting(float *const image[3], float *const residual_out[3], int *const mask[3],
                               const size_t width, const size_t height,
                               const int zut, const int iterations,
                               const float structure_threshold, const float update_speed)
{
  // Simultaneous inpainting for image structure and texture using anisotropic heat transfer model
  // https://www.researchgate.net/publication/220663968

  float A = structure_threshold / 100.0f;
  float B = (1.0f - A) * 0.25f;
  //A *= 0.1f;
  //B *= 0.1f;
  const float K = update_speed * 2.0f;

  float *const tmp = dt_alloc_sse_ps(width * height);

  for(size_t chan = 0; chan < 3; chan++)
  {
  // Initialize the masked area with random noise (plant seeds for diffusion)
  // From :
  //    Nontexture Inpainting by Curvature-Driven Diffusions
  //    Tony F. Chan, Jackie Jianhong Shen
  //    https://conservancy.umn.edu/bitstream/handle/11299/3528/1743.pdf?sequence=1
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(mask, tmp, image, height, width, chan, A) \
schedule(static) collapse(2) aligned(residual_out, image, tmp, mask:64)
#endif
    for(size_t i = 0; i < height; ++i)
      for(size_t j = 0; j < width; ++j)
      {
        const size_t index = i * width + j;
        tmp[index] = (mask[chan][index]) ? image[chan][index] + (0.5f - ((float)rand() / (float)RAND_MAX)) * 0.5f
                                         : image[chan][index];
      }

  // BEGINNING OF ITERATIONS

    int radius = 1;

    for(int iter = 0; iter < iterations; iter++)
    {
      radius += 1;
      if(radius > 24) radius = 1;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
dt_omp_firstprivate(tmp, mask, residual_out, height, width, chan, K, iter, A, B, radius) \
schedule(dynamic) collapse(2)
#endif
      for(size_t i = 2 + radius; i < height - 2 - radius; ++i)
        for(size_t j = 2 + radius; j < width - 2 - radius; ++j)
        {
          const size_t index = i * width + j;
          if(mask[chan][index])
          {
            float Delta_u_s = 0.0f;
            float Delta_u_t = 0.0f;

            // Discretization parameters for the Partial Derivative Equation solver
            const size_t h = 1;          // spatial step
            const float kappa = 0.25f; // depends on the spatial step
            const float det = kappa / sqf((float)h);

            const float u_n = tmp[index]; // center pixel u(i, j)

            const float u[8] DT_ALIGNED_PIXEL = { tmp[(i - h) * width + (j - h)],
                                                  tmp[(i) * width + (j - h)],
                                                  tmp[(i + h) * width + (j - h)],
                                                  tmp[(i - h) * width + (j)],
                                                  // tmp[(i) * width + (j)] : u(i, j) -> special handling
                                                  tmp[(i + h) * width + (j)],
                                                  tmp[(i - h) * width + (j + h)],
                                                  tmp[(i) * width + (j + h)],
                                                  tmp[(i + h) * width + (j + h)] };

            // Compute the gradient with centered finite differences
            const float grad_y = (tmp[(i + h) * width + j] - tmp[(i - h) * width + j]) / (2.0f * (float)h); // du(i, j) / dy
            const float grad_x = (tmp[(i) * width + (j + h)] - tmp[(i) * width + (j - h)]) / (2.0f * (float)h); // du(i, j) / dx

            // Find the direction of the gradient
            const float theta = atan2f(grad_y, grad_x);
            const float sin_theta = sinf(theta);
            const float sin_theta2 = sqf(sin_theta);
            const float cos_theta = cosf(theta);
            const float cos_theta2 = sqf(cos_theta);


            // Structure extraction : probably not needed for highlights recovery
            if(A > 0.0f)
            {
              // Find the dampening factor
              const float c = expf(-hypotf(grad_x, grad_y) / K);
              const float c2 = sqf(c);

              // Build the convolution kernel for the structure extraction
              const float a11 = cos_theta2 + c2 * sin_theta2;
              const float a12 = (c2 - 1.0f) * cos_theta * sin_theta;
              const float a22 = c2 * cos_theta2 + sin_theta2;

              const float b11 = (-a12 / 2.0f);
              const float b13 = -b11;

              const float kern[8] DT_ALIGNED_PIXEL = { b11, a22, b13,
                                                       a11,      a11, // b22 -> special handling
                                                       b13, a22, b11 };

              // Note : b22 and u(i, j) are handled apart and used in reductions initialization
              // so u and kern are full SSE3 vectors
              Delta_u_s += -2.0f * (a11 + a22) * u_n;

              // Convolve
#ifdef _OPENMP
#pragma omp simd aligned(kern, u:64) reduction(+:Delta_u_s)
#endif
              for(size_t ki = 0; ki < 8; ki++)
                Delta_u_s += kern[ki] * u[ki];

              Delta_u_s *= det;
            }

            // Texture extraction
            if(B > 0.0f)
            {
              // Find alpha, the principal direction of the grad(grad(u(i,j))
              // NOTE : derivating a 2D vector in the euclidian plane is equivalent to a +pi/2 rotation
              // we already have theta, the principal direction of the vector grad(u(i, j)) = { du / dx, du / dy },
              // so we know alpha = theta + pi / 2,
              // thus cos(alpha) = cos(theta + pi / 2) = -sin(theta)
              // and sin(alpha) = sin(theta + pi / 2) = cos(theta)
              // So we reuse cos(theta) and sin(theta) to avoid recomputing grad(grad()), cos and sin

              // Compute the corners of the texture searching window oriented in the principal
              // direction of the grad
              const int dx = roundf((float)radius * (-sin_theta));
              const int dy = roundf((float)radius * cos_theta);

              const float du_dzeta = (tmp[(i - dx) * width + (j + dy)]
                                    + tmp[(i + dx) * width + (j - dy)])
                                    - 2.0f * u_n;
              const float du_eta = (tmp[(i + dy) * width + (j + dx)]
                                  + tmp[(i - dy) * width + (j - dx)])
                                  - 2.0f * u_n;

              Delta_u_t += (du_dzeta + du_eta);
            }

            residual_out[chan][index] = A * Delta_u_s + B * Delta_u_t;
          }
        }


  // Update the current reconstructed image
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(tmp, mask, residual_out, height, width, chan, radius) \
schedule(static) collapse(2) aligned(tmp, mask, residual_out:64)
#endif
      for(size_t i = 2 + radius; i < height - 2 - radius; ++i)
        for(size_t j = 2 + radius; j < width - 2 - radius; ++j)
        {
          const size_t index = i * width + j;
          if(mask[chan][index])
            tmp[index] += residual_out[chan][index];
        }
    }

    // END OF ITERATIONS

    // Copy solution and cleanup
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(mask, tmp, image, height, width, chan, A) \
schedule(static) collapse(2) aligned(residual_out, tmp, mask, image:64)
#endif
    for(size_t i = 0; i < height; ++i)
      for(size_t j = 0; j < width; ++j)
      {
        const size_t index = i * width + j;
        image[chan][index] = (mask[chan][index]) ? tmp[index]
                                                 : image[chan][index];
      }
  }

  dt_free_align(tmp);

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

  int *const mask[3] = { dt_alloc_align(64, width * height * sizeof(int)),
                         dt_alloc_align(64, width * height * sizeof(int)),
                         dt_alloc_align(64, width * height * sizeof(int)) };

  float *const restrict luma = dt_alloc_sse_ps(width * height);
  float *const restrict luma_low_freq = dt_alloc_sse_ps(width * height);
  float *const restrict luma_high_freq = dt_alloc_sse_ps(width * height);

  const float epsilon = 100.0f;
  //const float luma_feathering = 1.f / powf(10.f, 2.0f * d->luma_feathering);
  //const float luma_strength = d->luma_strength /  epsilon;

  const float chroma_feathering = 1.f / powf(10.f, 2.0f * d->chroma_feathering);
  const float chroma_strength = d->chroma_strength / epsilon;

  const float fringes_feathering = 1.f / powf(10.f, 2.0f * d->fringes_feathering);
  const float fringes_strength = d->fringes_strength / epsilon;

  const float sharpness_feathering = 1.f / powf(10.f, 2.0f * d->sharpness_feathering);
  const float sharpness_strength = d->sharpness_strength / epsilon;

  const float highlight_clipping = d->highlight_clipping / 100.0f;
  //const float lowlight_clipping = exp2f(d->lowlight_clipping);

  const int run_chroma = (chroma_strength != 0.0f);
  //const int run_luma = (luma_strength != 0.0f);
  const int run_fringes = (fringes_strength != 0.0f);
  const int run_sharpness = (sharpness_strength != 0.0f);

  // Peel the array of structs into RGB layers (struct of arrays) and search for clipped pixels
  int mask_number = 0;

  const size_t i_max =  height / 2 + 20;
  const size_t i_min = height / 2 - 20;
  const size_t j_min = width / 2 - 10;
  const size_t j_max = width / 2 + 10;

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(out_RGB, in_RGB, mask, in, height, width, ch, highlight_clipping, i_min, i_max, j_min, j_max) \
schedule(static) collapse(3) reduction(+:mask_number) aligned(mask, out_RGB, in_RGB, in:64)
#endif
  for(size_t i = 0; i < height; ++i)
    for(size_t j = 0; j < width; ++j)
      for(size_t c = 0; c < 3; c++)
      {
        const size_t index = (i * width + j);
        const int masked = (FALSE && (i > i_min) && (i < i_max) && (j < j_max) && (j > j_min));
        in_RGB[c][index] = out_RGB[c][index] = in[index * ch + c];
        mask[c][index] = ((in[index * ch + c] > highlight_clipping) || masked);
        mask_number += mask[c][index];
      }

  // Apply a uniform blur on clipped areas to avoid sharp transitions between
  // non-clipped and clipped areas
  //for(size_t c = 0; c < 3; c++)
  // box_average(out_RGB[c], width, height, 1, MAX(1, d->scales / piece->iscale * roi_in->scale));

  // Restore data where it doesn't clip and add gaussian noise in clipped areas to simulate texture
  /*
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(out_RGB, in_RGB, mask, in, height, width) \
schedule(dynamic) collapse(3) reduction(+:mask_number) aligned(mask, out_RGB, in_RGB:64)
#endif
  for(size_t i = 0; i < height; ++i)
    for(size_t j = 0; j < width; ++j)
      for(size_t c = 0; c < 3; c++)
      {
        const size_t index = (i * width + j);
        if(mask[c][index])
          out_RGB[c][index] *= 1.0f - sqf((float)rand() / (float)RAND_MAX) / 10.0f;
        else
          out_RGB[c][index] = in_RGB[c][index];
      }
*/
  for(int k = 0; k < d->scales; k++)
  {
    // At each size iteration, we increase the width of the guided filter window
    const int radius = MAX(1, (k + 1) / piece->iscale * roi_in->scale);

    // At each size iteration, we decrease the width of the inpainting texture searching window
    const int distance = MAX(1, (d->iterations - k) / piece->iscale * roi_in->scale);

    float scale;
    if(radius % 4 == 0)
      scale = 4.0f;
    else if(radius % 3 == 0)
      scale = 3.0f;
    else if(radius % 2 == 0)
      scale = 2.0f;
    else
      scale = 1.f;

    if(mask_number > 0 && d->reconstruct_iterations > 0)
      heat_PDE_inpanting(out_RGB, residual_out, mask, width, height, distance,
                            d->reconstruct_iterations, d->structure_threshold, d->update_speed);


    for(int outer_iter = 0; outer_iter < d->iterations; outer_iter++)
    {
      if(run_chroma)
        denoise_chroma(out_RGB, residual_out, width, height, scale, radius, chroma_feathering, chroma_strength);

      if(run_fringes)
      {
        // Inspired by
        // MULTISPECTRAL DEMOSAICKING WITH NOVEL GUIDE IMAGE GENERATIONAND RESIDUAL INTERPOLATION
        // Yusuke Monno, Daisuke Kiku, Sunao Kikuchi, Masayuki Tanaka, and Masatoshi Okutomi
        // https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICIP-2014/Papers/1569909365.pdf
        for(size_t c = 0; c < 3; c++)
          fast_guided_filter_rgb(out_RGB[c], out_RGB[1], residual_out[c], width, height, 1, radius, fringes_feathering, scale, TRUE);

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(out_RGB, RGB_high_freq, residual_out, luma, height, width, ch, fringes_strength) \
schedule(static) collapse(3)
#endif
        for(size_t c = 0; c < 3; c++)
        {
          for(size_t i = 0; i < height; ++i)
          {
            for(size_t j = 0; j < width; ++j)
            {
              const size_t index = i * width + j;
              RGB_high_freq[c][index] = out_RGB[c][index] - residual_out[c][index];
            }
          }
        }

        for(size_t c = 0; c < 3; c++)
          fast_guided_filter_rgb(RGB_high_freq[c], RGB_high_freq[1], residual_out[c], width, height, 1, radius, fringes_feathering, scale, FALSE);

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(out_RGB, RGB_high_freq, residual_out, luma, height, width, ch, fringes_strength) \
schedule(static) collapse(3)
#endif
        for(size_t c = 0; c < 3; c++)
        {
          for(size_t i = 0; i < height; ++i)
          {
            for(size_t j = 0; j < width; ++j)
            {
              const size_t index = i * width + j;
              out_RGB[c][index] += fringes_strength * residual_out[c][index];
            }
          }
        }
      }

      if(run_sharpness)
      {
        for(int c = 0; c < 3; c++)
          fast_guided_filter_rgb(out_RGB[c], out_RGB[c], residual_out[c], width, height, 1, radius, sharpness_feathering, scale, FALSE);

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
    dt_free_align(mask[c]);
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
                                                         .iterations = 1,
                                                         .luma_strength = 15.f,
                                                         .luma_feathering = 2.5f,
                                                         .chroma_strength = 50.f,
                                                         .chroma_feathering = 2.0f, // chroma
                                                         .fringes_strength = 10.f,
                                                         .fringes_feathering = 1.5f, // fringes
                                                         .sharpness_strength = 10.f,
                                                         .sharpness_feathering = 1.5f,   // sharpness
                                                         .highlight_clipping = 99.0f,
                                                         .structure_threshold = 0.0f,
                                                         .update_speed = 1.0f,
                                                         .reconstruct_iterations = 90,
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
  dt_bauhaus_slider_set_soft(g->structure_threshold, p->structure_threshold);
  dt_bauhaus_slider_set_soft(g->update_speed, p->update_speed);
  dt_bauhaus_slider_set_soft(g->reconstruct_iterations, p->reconstruct_iterations);

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

static void structure_threshold_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->structure_threshold = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void update_speed_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->update_speed = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void reconstruct_iterations_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->reconstruct_iterations = dt_bauhaus_slider_get(slider);
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

  g->highlight_clipping = dt_bauhaus_slider_new_with_range(self, 0., 150., 1., 100., 2);
  dt_bauhaus_slider_set_format(g->highlight_clipping, "%.2f %%");
  dt_bauhaus_widget_set_label(g->highlight_clipping, NULL, _("highlights clipping"));
  gtk_box_pack_start(GTK_BOX(page4), g->highlight_clipping, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->highlight_clipping), "value-changed", G_CALLBACK(highlight_clipping_callback), self);

  g->lowlight_clipping = dt_bauhaus_slider_new_with_range(self, -16., 0., 1., -12., 2);
  dt_bauhaus_slider_set_format(g->lowlight_clipping, "%.2f EV");
  dt_bauhaus_widget_set_label(g->lowlight_clipping, NULL, _("lowlights clipping"));
  gtk_box_pack_start(GTK_BOX(page4), g->lowlight_clipping, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->lowlight_clipping), "value-changed", G_CALLBACK(lowlight_clipping_callback), self);

  g->structure_threshold = dt_bauhaus_slider_new_with_range(self, 0., 100., 1., 75.0f, 2);
  dt_bauhaus_slider_set_format(g->structure_threshold, "%.2f %%");
  dt_bauhaus_widget_set_label(g->structure_threshold, NULL, _("structure vs. texture threshold"));
  gtk_box_pack_start(GTK_BOX(page4), g->structure_threshold, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->structure_threshold), "value-changed", G_CALLBACK(structure_threshold_callback), self);

  g->update_speed = dt_bauhaus_slider_new_with_range(self, 1., 20., 1., 1.0f, 2);
  dt_bauhaus_widget_set_label(g->update_speed, NULL, _("smoothing"));
  gtk_box_pack_start(GTK_BOX(page4), g->update_speed, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->update_speed), "value-changed", G_CALLBACK(update_speed_callback), self);

  g->reconstruct_iterations = dt_bauhaus_slider_new_with_range(self, 0, 600, 1, 50, 0);
  dt_bauhaus_widget_set_label(g->reconstruct_iterations, NULL, _("reconstruction iterations"));
  gtk_box_pack_start(GTK_BOX(page4), g->reconstruct_iterations, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->reconstruct_iterations), "value-changed", G_CALLBACK(reconstruct_iterations_callback), self);

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
