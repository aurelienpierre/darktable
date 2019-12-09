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
  float luma_strength, luma_feathering, luma_offset;
  float chroma_strength, chroma_feathering, chroma_offset;
  float fringes_strength, fringes_feathering, fringes_offset, fringes_regularization;
  float sharpness_strength, sharpness_feathering, sharpness_offset, sharpness_width, dof_sensitivity;
  float highlight_clipping, lowlight_clipping, structure_threshold, update_speed;
  int reconstruct_iterations;
} dt_iop_doctor_params_t;

typedef struct dt_iop_doctor_gui_data_t
{
  GtkNotebook *notebook;
  GtkWidget *scales, *iterations;
  GtkWidget *luma_feathering, *luma_strength, *luma_offset;
  GtkWidget *chroma_feathering, *chroma_strength, *chroma_offset;
  GtkWidget *fringes_feathering, *fringes_strength, *fringes_offset, *fringes_regularization;
  GtkWidget *highlight_clipping, *lowlight_clipping;
  GtkWidget *sharpness_feathering, *sharpness_strength, *sharpness_offset, *sharpness_width, *dof_sensitivity;
  GtkWidget *structure_threshold, *update_speed, *reconstruct_iterations;
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


#ifdef _OPENMP
#pragma omp declare simd aligned(image:64) uniform(image)
#endif
static inline float total_variation(const float *const image, const size_t index[5], const float eps)
{
  return 2.0f * (-4.0f * image[index[0]] + image[index[1]] + image[index[2]] + image[index[3]] + image[index[4]]) /
                  (hypotf(image[index[0]] - image[index[1]], image[index[0]] - image[index[3]]) +
                    hypotf(image[index[2]] - image[index[0]], image[index[4]] - image[index[0]]) + eps);
}


#ifdef _OPENMP
#pragma omp declare simd
#endif
static inline void get_indices(const size_t i, const size_t j, const size_t width, const size_t height, size_t index[5])
{
  index[0] = i * width + j;       // center
  index[1] = (i - 1) * width + j; // north
  index[2] = (i + 1) * width + j; // south
  index[3] = i * width + j - 1;   // west
  index[4] = i * width + j + 1;   // east
}


static inline void denoise_chroma(float *const image[3], float *const temp_buffer[4],
                                  const size_t width, const size_t height,
                                  const int radius, const float scale,
                                  const float chroma_feathering, const float chroma_strength)
{
  float b = chroma_strength;

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(image, temp_buffer, height, width) \
schedule(static) collapse(2) aligned(temp_buffer, image:64)
#endif
  for(size_t i = 0; i < height; ++i)
    for(size_t j = 0; j < width; ++j)
    {
      const size_t index = i * width + j;
      temp_buffer[0][index] = image[1][index] - image[0][index]; // G - R
      temp_buffer[1][index] = image[1][index] - image[2][index]; // G - B
    }

  fast_guided_filter_rgb(temp_buffer[0], temp_buffer[0], temp_buffer[2], width, height, 1, radius, chroma_feathering, scale, FALSE);
  fast_guided_filter_rgb(temp_buffer[1], temp_buffer[1], temp_buffer[3], width, height, 1, radius, chroma_feathering, scale, FALSE);

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(image, temp_buffer, height, width, b) \
schedule(static) collapse(2) aligned(temp_buffer, image:64)
#endif
  for(size_t i = 0; i < height; ++i)
    for(size_t j = 0; j < width; ++j)
    {
      const size_t index = i * width + j;

      const float GR = temp_buffer[0][index]; // G - R
      const float GB = temp_buffer[1][index]; // G - B

      const float LF_GR = temp_buffer[2][index];
      const float LF_GB = temp_buffer[3][index];

      const float HF_GR = b * (GR - LF_GR);
      const float HF_GB = b * (GB - LF_GB);

      image[0][index] += HF_GR;
      image[1][index] -= (HF_GR + HF_GB) / 2.0f;
      image[2][index] += HF_GB;
    }

  fast_guided_filter_rgb(temp_buffer[2], temp_buffer[2], temp_buffer[0], width, height, 1, radius, chroma_feathering, scale, FALSE);
  fast_guided_filter_rgb(temp_buffer[3], temp_buffer[3], temp_buffer[1], width, height, 1, radius, chroma_feathering, scale, FALSE);

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(image, temp_buffer, height, width, b) \
schedule(static) collapse(2) aligned(temp_buffer, image:64)
#endif
  for(size_t i = 0; i < height; ++i)
    for(size_t j = 0; j < width; ++j)
    {
      const size_t index = i * width + j;

      const float GR = temp_buffer[2][index]; // G - R
      const float GB = temp_buffer[3][index]; // G - B

      const float LF_GR = temp_buffer[0][index];
      const float LF_GB = temp_buffer[1][index];

      const float HF_GR = b * (GR - LF_GR);
      const float HF_GB = b * (GB - LF_GB);

      image[0][index] += HF_GR;
      image[1][index] -= (HF_GR + HF_GB) / 2.0f;
      image[2][index] += HF_GB;
    }
}


static inline void denoise_chroma_crossed(float *const image[3], float *const temp_buffer[4],
                                          const size_t width, const size_t height,
                                          const int radius, const float scale,
                                          const float chroma_feathering, const float chroma_strength)
{
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(image, temp_buffer, height, width) \
schedule(static) collapse(2) aligned(temp_buffer, image:64)
#endif
  for(size_t i = 0; i < height; ++i)
    for(size_t j = 0; j < width; ++j)
    {
      const size_t index = i * width + j;
      temp_buffer[0][index] = image[1][index] - image[0][index]; // G - R
      temp_buffer[1][index] = image[1][index] - image[2][index]; // G - B
    }

  fast_guided_filter_rgb(temp_buffer[0], temp_buffer[1], temp_buffer[2], width, height, 1, radius, chroma_feathering, scale, FALSE);
  fast_guided_filter_rgb(temp_buffer[1], temp_buffer[0], temp_buffer[3], width, height, 1, radius, chroma_feathering, scale, FALSE);

  const float b = chroma_strength / 2.0f;

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(image, temp_buffer, height, width, b) \
schedule(static) collapse(2) aligned(temp_buffer, image:64)
#endif
  for(size_t i = 0; i < height; ++i)
    for(size_t j = 0; j < width; ++j)
    {
      const size_t index = i * width + j;

      const float GR = temp_buffer[0][index]; // G - R
      const float GB = temp_buffer[1][index]; // G - B

      const float LF_GR = temp_buffer[2][index];
      const float LF_GB = temp_buffer[3][index];

      const float HF_GR = b * (GR - LF_GR);
      const float HF_GB = b * (GB - LF_GB);

      image[0][index] += HF_GR;
      image[1][index] -= (HF_GR + HF_GB) / 2.0f;
      image[2][index] += HF_GB;
    }
}

static void heat_PDE_inpanting(float *const image[3], float *const mask[3],
                               const size_t width, const size_t height,
                               const int iterations,
                               const float structure_threshold, const float update_speed)
{
  // Simultaneous inpainting for image structure and texture using anisotropic heat transfer model
  // https://www.researchgate.net/publication/220663968

  // Discretization parameters for the Partial Derivative Equation solver
  const size_t h = 1;          // spatial step
  const float kappa = 0.25f;    // 0.25 if h = 1, 1 if h = 2

  float A = structure_threshold / 100.0f;
  float B = (1.0f - A) * 0.25f;
  A *= update_speed * kappa;
  B *= update_speed;
  const float K = 2.0f;

  const int run_structure = (A > 0.0f);
  const int run_texture = (B > 0.0f);

  float *const tmp = dt_alloc_sse_ps(width * height);
  float *const update = dt_alloc_sse_ps(width * height);

  for(int chan = 0; chan < 3; ++chan)
  {
    // Initialize the masked area with random noise (plant seeds for diffusion)
    // From :
    //    Nontexture Inpainting by Curvature-Driven Diffusions
    //    Tony F. Chan, Jackie Jianhong Shen
    //    https://conservancy.umn.edu/bitstream/handle/11299/3528/1743.pdf?sequence=1
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(mask, image, tmp, height, width, chan) \
schedule(static) collapse(2) aligned(tmp, mask, image:64)
#endif
    for(size_t i = 0; i < height; ++i)
      for(size_t j = 0; j < width; ++j)
      {
        const size_t index = i * width + j;
        tmp[index] = image[chan][index] + mask[chan][index] * (0.5f - ((float)rand() / (float)RAND_MAX)) * 0.5f;
      }

    // BEGINNING OF ITERATIONS
    for(int iter = 0; iter < iterations; iter++)
    {
      //const int radius = 2 + (iter % 2) + (iter % 4) + (iter % 8);
      const size_t radius = 1;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
dt_omp_firstprivate(tmp, update, mask, height, width, K, iter, A, B, radius, run_structure, run_texture, h, chan) \
schedule(dynamic) collapse(2)
#endif
      for(size_t i = 2 + radius; i < height - 2 - radius; ++i)
        for(size_t j = 2 + radius; j < width - 2 - radius; ++j)
        {
          const size_t index = i * width + j;
          float update_index = 0.0f;

          if(mask[chan][index] > 0.0f)
          {
            // Cache the useful pixels
            const float u_n = tmp[index]; // center pixel u(i, j)

            // neighbours
            const size_t j_prev = j - h; // x
            const size_t j_next = j + h; // x
            const size_t j_spot = j;
            const size_t i_prev = (i - h) * width; // y
            const size_t i_next = (i + h) * width; // y
            const size_t i_spot = i * width;
            const float u[8] DT_ALIGNED_PIXEL = { tmp[i_prev + j_prev], tmp[i_prev + j_spot], tmp[i_prev + j_next],
                                                  tmp[i_spot + j_prev],                       tmp[i_spot + j_next],
                                                  tmp[i_next + j_prev], tmp[i_next + j_spot], tmp[i_next + j_next] };

            // Compute the gradient with centered finite differences
            const float grad_y = (u[4] - u[5]) / 2.0f; // du(i, j) / dy
            const float grad_x = (u[6] - u[1]) / 2.0f; // du(i, j) / dx

            // Find the direction of the gradient
            const float theta = atan2f(grad_y, grad_x);
            const float sin_theta = sinf(theta);
            const float cos_theta = cosf(theta);

            // Structure extraction : probably not needed for highlights recovery
            if(run_structure)
            {
              const float sin_theta2 = sqf(sin_theta);
              const float cos_theta2 = sqf(cos_theta);

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
              float Delta_u_s = -2.0f * (a11 + a22) * u_n;

              // Convolve
#ifdef _OPENMP
#pragma omp simd aligned(kern, u:64) reduction(+:Delta_u_s)
#endif
              for(size_t ki = 0; ki < 8; ki++)
                Delta_u_s += kern[ki] * u[ki];

              update_index += A * Delta_u_s;
            }

            // Texture extraction
            if(run_texture)
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
              const int dx = (int)CLAMP((int)(-(float)radius * sin_theta), -radius, radius);
              const int dy = (int)CLAMP((int)((float)radius * cos_theta), -radius, radius);

              const float du_dzeta = (tmp[(i - dx) * width + (j + dy)]
                                    + tmp[(i + dx) * width + (j - dy)])
                                    - 2.0f * u_n;
              const float du_eta = (tmp[(i + dy) * width + (j + dx)]
                                  + tmp[(i - dy) * width + (j - dx)])
                                  - 2.0f * u_n;

              update_index += B * (du_dzeta + du_eta);
            }
          }

          update[index] = update_index;

        }

  // Update the current reconstructed image with alpha blending
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(tmp, mask, update, height, width, radius, chan) \
schedule(static) collapse(2) aligned(tmp, mask, update:64)
#endif
      for(size_t i = 2 + radius; i < height - 2 - radius; ++i)
        for(size_t j = 2 + radius; j < width - 2 - radius; ++j)
        {
          const size_t index = i * width + j;
          tmp[index] += mask[chan][index] * update[index];
        }
    }

    // END OF ITERATIONS

    // Copy solution with alpha blending
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(mask, tmp, image, height, width, chan) \
schedule(static) collapse(2) aligned(tmp, mask, image:64)
#endif
      for(size_t i = 0; i < height; ++i)
        for(size_t j = 0; j < width; ++j)
        {
          const size_t index = i * width + j;
          image[chan][index] = mask[chan][index] * tmp[index] + (1.0f - mask[chan][index]) * image[chan][index];
        }
  }

  dt_free_align(tmp);
  dt_free_align(update);
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

  float *const mask[3] = { dt_alloc_sse_ps(width * height),
                           dt_alloc_sse_ps(width * height),
                           dt_alloc_sse_ps(width * height) };

  float *const restrict luma = dt_alloc_sse_ps(width * height);
  float *const restrict luma_low_freq = dt_alloc_sse_ps(width * height);
  float *const restrict luma_high_freq = dt_alloc_sse_ps(width * height);

  const float epsilon = 100.0f;

  const float fringes_first_order = powf(10.f, -2.0f * d->fringes_regularization);
  const float fringes_second_order = powf(10.f, -2.0f * d->fringes_strength);

  const float sharpness_strength = d->sharpness_strength / epsilon;
  const float chroma_strength = d->chroma_strength / epsilon;
  const float luma_strength = d->luma_strength / epsilon;

  const float highlight_clipping = d->highlight_clipping / 100.0f;
  const float lowlight_clipping = exp2f(d->lowlight_clipping);

  const int run_chroma = (chroma_strength != 0.0f);
  const int run_luma = (luma_strength != 0.0f);
  const int run_fringes = TRUE;
  const int run_sharpness = (sharpness_strength != 0.0f);

  // Peel the array of structs into RGB layers (struct of arrays)
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(out_RGB, in_RGB, in, height, width, ch, piece) \
schedule(static) collapse(3) aligned(out_RGB, in_RGB, in:64)
#endif
  for(size_t i = 0; i < height; ++i)
    for(size_t j = 0; j < width; ++j)
      for(size_t c = 0; c < 3; c++)
      {
        const size_t index = (i * width + j);
        in_RGB[c][index] = out_RGB[c][index] = in[index * ch + c];
      }

  const float sharpness_step = (d->scales > 1) ? (d->sharpness_offset - d->sharpness_feathering) / ((float)d->scales - 1.0f) : 0.0f;

  for(int k = 0; k < d->scales; k++)
  {
    const float luma_feathering = powf(10.f, -2.0f * d->luma_feathering);
    const float chroma_feathering = powf(10.f, -2.0f * (d->chroma_feathering + k * d->chroma_offset));
    const float fringes_feathering = powf(10.f, -2.0f * (d->fringes_feathering + k * d->fringes_offset));
    const float sharpness_feathering_low = powf(10.f, -2.0f * (d->sharpness_feathering + k * sharpness_step));
    const float sharpness_denoise = powf(10.f, -2.0f * d->sharpness_width);

    // At each size iteration, we increase the width of the guided filter window
    const int radius = MAX(1, (k + 1) / piece->iscale * roi_in->scale);

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

      if(run_fringes)
      {
        // Inspired by
        // MULTISPECTRAL DEMOSAICKING WITH NOVEL GUIDE IMAGE GENERATIONAND RESIDUAL INTERPOLATION
        // Yusuke Monno, Daisuke Kiku, Sunao Kikuchi, Masayuki Tanaka, and Masatoshi Okutomi
        // https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICIP-2014/Papers/1569909365.pdf

        // Get the RGB low frequencies
        fast_guided_filter_rgb(out_RGB[0], out_RGB[1], residual_out[0], width, height, 1, radius, fringes_feathering, scale, TRUE);
        fast_guided_filter_rgb(out_RGB[2], out_RGB[1], residual_out[2], width, height, 1, radius, fringes_feathering, scale, TRUE);
        fast_guided_filter_rgb(out_RGB[1], out_RGB[1], residual_out[1], width, height, 1, radius, fringes_feathering, scale, TRUE);

        // Get the RGB high frequencies
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(out_RGB, RGB_high_freq, residual_out, luma, height, width, ch) \
schedule(static) collapse(3) aligned(out_RGB, RGB_high_freq, residual_out:64)
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

        // Remove gradients (first order)
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(out_RGB, RGB_high_freq, residual_out, luma_high_freq, height, width, ch, fringes_first_order) \
schedule(static) collapse(2) aligned(out_RGB, RGB_high_freq, luma_high_freq, residual_out:64)
#endif
        for(size_t i = 0; i < height; ++i)
          for(size_t j = 0; j < width; ++j)
          {
            const size_t index = i * width + j;

            const float norm = 1.0f;// / sqrtf( sqf(RGB_high_freq[0][index]) + sqf(RGB_high_freq[1][index]) + sqf(RGB_high_freq[2][index]));

            // Solve u = u + d(grad(u)) / d nu + lambda × TV in R
            out_RGB[0][index] -= fringes_first_order * (RGB_high_freq[1][index] - RGB_high_freq[0][index]) * norm;

            // Solve u = u + d(grad(u)) / d nu + lambda × TV in B
            out_RGB[2][index] -= fringes_first_order * (RGB_high_freq[1][index] - RGB_high_freq[2][index]) * norm;

            // Solve Solve u = u - d²(grad(u)) / d nu² + lambda × TV in G
            out_RGB[1][index] += fringes_first_order * (RGB_high_freq[0][index] + RGB_high_freq[2][index] - 2.0f * RGB_high_freq[1][index]) * norm;
          }

        // Detect fringes - equivalent to a laplacian filter with edge-awareness
        fast_guided_filter_rgb(RGB_high_freq[0], RGB_high_freq[1], residual_out[0], width, height, 1, radius, fringes_feathering, scale, FALSE);
        fast_guided_filter_rgb(RGB_high_freq[2], RGB_high_freq[1], residual_out[2], width, height, 1, radius, fringes_feathering, scale, FALSE);
        fast_guided_filter_rgb(RGB_high_freq[1], RGB_high_freq[1], residual_out[1], width, height, 1, radius, fringes_feathering, scale, FALSE);

        // Get the RGB high frequencies
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(out_RGB, RGB_high_freq, residual_out, luma, height, width, ch) \
schedule(static) collapse(3) aligned(out_RGB, RGB_high_freq, residual_out:64)
#endif
        for(size_t c = 0; c < 3; c++)
        {
          for(size_t i = 0; i < height; ++i)
          {
            for(size_t j = 0; j < width; ++j)
            {
              const size_t index = i * width + j;
              RGB_high_freq[c][index] = RGB_high_freq[c][index] - residual_out[c][index];
            }
          }
        }

        // Remove fringes (second order)
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(out_RGB, RGB_high_freq, residual_out, luma_high_freq, height, width, ch, fringes_second_order) \
schedule(static) collapse(2) aligned(out_RGB, RGB_high_freq, luma_high_freq, residual_out:64)
#endif
        for(size_t i = 0; i < height; ++i)
          for(size_t j = 0; j < width; ++j)
          {
            const size_t index = i * width + j;

            const float norm = 1.0f;// / sqrtf( sqf(RGB_high_freq[0][index]) + sqf(RGB_high_freq[1][index]) + sqf(RGB_high_freq[2][index]));

            // Solve u = u + d(grad(u)) / d nu + lambda × TV in R
            out_RGB[0][index] -= fringes_second_order * (RGB_high_freq[1][index] - RGB_high_freq[0][index]) * norm;

            // Solve u = u + d(grad(u)) / d nu + lambda × TV in B
            out_RGB[2][index] -= fringes_second_order * (RGB_high_freq[1][index] - RGB_high_freq[2][index]) * norm;

            // Solve Solve u = u - d²(grad(u)) / d nu² + lambda × TV in G
            out_RGB[1][index] += fringes_second_order * (RGB_high_freq[0][index] + RGB_high_freq[2][index] - 2.0f * RGB_high_freq[1][index]) * norm;
          }

      }


      if(run_chroma)
        denoise_chroma(out_RGB, residual_out, width, height, scale, radius, chroma_feathering, chroma_strength);


      if(run_luma)
        denoise_chroma_crossed(out_RGB, residual_out, width, height, scale, radius, luma_feathering, luma_strength);


      if(run_sharpness)
      {
        const float dof = d->dof_sensitivity / 100.0f;

        for(int c = 0; c < 3; c++)
        {
          // Get the low pass 1 and lowpass 2,
          fast_guided_filter_rgb(out_RGB[c], out_RGB[c], residual_out[c], width, height, 1, radius, sharpness_feathering_low, scale, FALSE);
          //fast_guided_filter_rgb(out_RGB[c], out_RGB[c], RGB_high_freq[c], width, height, 1, radius, sharpness_feathering_high, scale, FALSE);

          // Compute the band-pass filter between the 2 corresponding high-pass filters.
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in_RGB, out_RGB, RGB_high_freq, residual_out, luma_high_freq, height, width, ch, sharpness_strength, c) \
schedule(static) collapse(2) aligned(in_RGB, out_RGB, residual_out, RGB_high_freq, luma_high_freq:64)
#endif
          for(size_t i = 0; i < height; ++i)
            for(size_t j = 0; j < width; ++j)
            {
              const size_t index = i * width + j;
              in_RGB[c][index] = out_RGB[c][index] - residual_out[c][index];
            }

          // Get the alpha blending mask
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(out_RGB, RGB_high_freq, luma_high_freq, residual_out, in_RGB, luma, mask, height, width, ch, sharpness_strength, c, dof) \
schedule(static) collapse(2) aligned(out_RGB, residual_out, RGB_high_freq, luma, mask, in_RGB:64)
#endif
          for(size_t i = 0; i < height; ++i)
            for(size_t j = 0; j < width; ++j)
            {
              const size_t index = i * width + j;
              mask[c][index] = clamp(dof * (0.5f + sqf(in_RGB[c][index])) / 2.0f);
            }

          fast_guided_filter_rgb(mask[c], mask[c], mask[c], width, height, 1, radius, sharpness_feathering_low, scale, TRUE);
        }

        // Apply the unsharp masking
#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(in_RGB, out_RGB, RGB_high_freq, residual_out, luma_high_freq, mask, luma, height, width, sharpness_strength, sharpness_denoise, radius) \
schedule(static) collapse(2) aligned(out_RGB, in_RGB, RGB_high_freq, luma_high_freq, residual_out, luma, mask:64)
#endif
          for(size_t i = 1; i < height - 1; ++i)
            for(size_t j = 1; j < width - 1; ++j)
            {
              size_t index[5];
              get_indices(i, j, width, height, index);

              const float eps = exp2f(-16.0f);

              const float TV[3] = { total_variation(out_RGB[0], index, eps),
                                    total_variation(out_RGB[1], index, eps),
                                    total_variation(out_RGB[2], index, eps) };

              const float TV_max = fmaxf(fmaxf(fabsf(TV[0]), fabsf(TV[1])), fabsf(TV[2]));
              const float RGB_min = fminf(fminf(out_RGB[0][index[0]], out_RGB[1][index[0]]), out_RGB[2][index[0]]);
              const float normalize = sharpness_denoise * RGB_min / TV_max;
              const float divTV = normalize * (TV[0] + TV[1] + TV[2]) / sqrtf(sqf(TV[0]) + sqf(TV[1]) + sqf(TV[2]) + eps);

              for(size_t c = 0; c < 3; c++)
                out_RGB[c][index[0]] += sharpness_strength * mask[c][index[0]] * (in_RGB[c][index[0]] + divTV);
            }

      }
    }
  }

  if(d->reconstruct_iterations > 0)
  {
    // limit to 80 iterations if we are in GUI to limit latency
    int iter = (self->dev->gui_attached && d->reconstruct_iterations > 80) ? 80 : d->reconstruct_iterations;

#ifdef _OPENMP
#pragma omp parallel for simd default(none) \
dt_omp_firstprivate(out_RGB, mask, height, width, highlight_clipping, lowlight_clipping, piece) \
schedule(static) collapse(3) aligned(mask, out_RGB, in:64)
#endif
    for(size_t c = 0; c < 3; c++)
      for(size_t i = 0; i < height; ++i)
        for(size_t j = 0; j < width; ++j)
        {
          // Build the clipping mask
          const size_t index = (i * width + j);
          mask[c][index] = (float)(out_RGB[c][index] > highlight_clipping || out_RGB[c][index] < lowlight_clipping);
        }

    // Blur the mask
    //box_average(mask[c], width, height, 1, 1);

    heat_PDE_inpanting(out_RGB, mask, width, height, iter, d->structure_threshold, d->update_speed / 100.0f);
  }

  // Repack the RGB layers to an array of stucts
#ifdef _OPENMP
#pragma omp parallel for default(none) \
dt_omp_firstprivate(out_RGB, out, height, width, ch, piece) \
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
                                                         .iterations = 3,
                                                         .luma_strength = 15.f,
                                                         .luma_feathering = 2.5f,
                                                         .luma_offset = 0.0f,
                                                         .chroma_strength = 50.f,
                                                         .chroma_feathering = 2.0f,
                                                         .chroma_offset = 0.5f,
                                                         .fringes_strength = 25.f,
                                                         .fringes_feathering = 1.0f,
                                                         .fringes_offset = -0.25f,
                                                         .fringes_regularization = 3.0f,
                                                         .sharpness_strength = 50.f,
                                                         .sharpness_feathering = 1.5f,
                                                         .sharpness_width = 0.15f,
                                                         .sharpness_offset = 2.5f,
                                                         .dof_sensitivity = 100.0f,
                                                         .highlight_clipping = 99.0f,
                                                         .structure_threshold = 50.0f,
                                                         .update_speed = 100.0f,
                                                         .reconstruct_iterations = 40,
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
  dt_bauhaus_slider_set_soft(g->luma_offset, p->luma_offset);

  dt_bauhaus_slider_set_soft(g->chroma_feathering, p->chroma_feathering);
  dt_bauhaus_slider_set_soft(g->chroma_strength, p->chroma_strength);
  dt_bauhaus_slider_set_soft(g->chroma_offset, p->chroma_offset);

  dt_bauhaus_slider_set_soft(g->fringes_feathering, p->fringes_feathering);
  dt_bauhaus_slider_set_soft(g->fringes_strength, p->fringes_strength);
  dt_bauhaus_slider_set_soft(g->fringes_offset, p->fringes_offset);
  dt_bauhaus_slider_set_soft(g->fringes_regularization, p->fringes_regularization);

  dt_bauhaus_slider_set_soft(g->highlight_clipping, p->highlight_clipping);
  dt_bauhaus_slider_set_soft(g->lowlight_clipping, p->lowlight_clipping);
  dt_bauhaus_slider_set_soft(g->structure_threshold, p->structure_threshold);
  dt_bauhaus_slider_set_soft(g->update_speed, p->update_speed);
  dt_bauhaus_slider_set_soft(g->reconstruct_iterations, p->reconstruct_iterations);

  dt_bauhaus_slider_set_soft(g->sharpness_feathering, p->sharpness_feathering);
  dt_bauhaus_slider_set_soft(g->sharpness_strength, p->sharpness_strength);
  dt_bauhaus_slider_set_soft(g->sharpness_offset, p->sharpness_offset);
  dt_bauhaus_slider_set_soft(g->sharpness_width, p->sharpness_width);
  dt_bauhaus_slider_set_soft(g->dof_sensitivity, p->dof_sensitivity);
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

static void luma_offset_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->luma_offset = dt_bauhaus_slider_get(slider);
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

static void chroma_offset_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->chroma_offset = dt_bauhaus_slider_get(slider);
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

static void fringes_regularization_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->fringes_regularization = dt_bauhaus_slider_get(slider);
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

static void fringes_offset_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->fringes_offset = dt_bauhaus_slider_get(slider);
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

static void sharpness_offset_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->sharpness_offset = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void sharpness_width_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->sharpness_width = dt_bauhaus_slider_get(slider);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void dof_sensitivity_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(self->dt->gui->reset) return;
  dt_iop_doctor_params_t *p = (dt_iop_doctor_params_t *)self->params;
  p->dof_sensitivity = dt_bauhaus_slider_get(slider);
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

  g->luma_offset = dt_bauhaus_slider_new_with_range(self, -1.0, 1.0, 0.5, 100., 2);
  dt_bauhaus_slider_set_format(g->luma_offset, "%.2f dB");
  dt_bauhaus_widget_set_label(g->luma_offset, NULL, _("offset between iterations"));
  gtk_box_pack_start(GTK_BOX(page1), g->luma_offset , FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->luma_offset), "value-changed", G_CALLBACK(luma_offset_callback), self);

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

  g->chroma_offset = dt_bauhaus_slider_new_with_range(self, -1.0, 1.0, 0.5, 100., 2);
  dt_bauhaus_slider_set_format(g->chroma_offset, "%.2f dB");
  dt_bauhaus_widget_set_label(g->chroma_offset, NULL, _("offset between iterations"));
  gtk_box_pack_start(GTK_BOX(page2), g->chroma_offset , FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->chroma_offset), "value-changed", G_CALLBACK(chroma_offset_callback), self);

  g->fringes_feathering = dt_bauhaus_slider_new_with_range(self, 0.1, 3., 0.2, 5., 2);
  dt_bauhaus_slider_set_format(g->fringes_feathering, "%.2f dB");
  dt_bauhaus_widget_set_label(g->fringes_feathering, NULL, _("edges sensitivity"));
  gtk_box_pack_start(GTK_BOX(page3), g->fringes_feathering, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->fringes_feathering), "value-changed", G_CALLBACK(fringes_feathering_callback), self);

  g->fringes_regularization = dt_bauhaus_slider_new_with_range(self, 0.0, 3., 0.05, 1.5, 2);
  dt_bauhaus_slider_set_format(g->fringes_regularization, "%.2f dB");
  dt_bauhaus_widget_set_label(g->fringes_regularization, NULL, _("gradient"));
  gtk_box_pack_start(GTK_BOX(page3), g->fringes_regularization, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->fringes_regularization), "value-changed", G_CALLBACK(fringes_regularization_callback), self);

  g->fringes_strength = dt_bauhaus_slider_new_with_range(self, 0.0, 3., 0.05, 1.5, 2);
  dt_bauhaus_slider_set_format(g->fringes_strength, "%.2f dB");
  dt_bauhaus_widget_set_label(g->fringes_strength , NULL, _("laplacian"));
  gtk_box_pack_start(GTK_BOX(page3), g->fringes_strength, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->fringes_strength), "value-changed", G_CALLBACK(fringes_strength_callback), self);

  g->fringes_offset = dt_bauhaus_slider_new_with_range(self, -1.0, 1.0, 0.5, 100., 2);
  dt_bauhaus_slider_set_format(g->fringes_offset, "%.2f dB");
  dt_bauhaus_widget_set_label(g->fringes_offset, NULL, _("offset between iterations"));
  gtk_box_pack_start(GTK_BOX(page3), g->fringes_offset , FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->fringes_offset), "value-changed", G_CALLBACK(fringes_offset_callback), self);

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

  g->update_speed = dt_bauhaus_slider_new_with_range(self, 0., 100., 1., 100.0f, 2);
  dt_bauhaus_slider_set_format(g->update_speed, "%.2f %%");
  dt_bauhaus_widget_set_label(g->update_speed, NULL, _("strength"));
  gtk_box_pack_start(GTK_BOX(page4), g->update_speed, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->update_speed), "value-changed", G_CALLBACK(update_speed_callback), self);

  g->reconstruct_iterations = dt_bauhaus_slider_new_with_range(self, 0, 600, 1, 50, 0);
  dt_bauhaus_widget_set_label(g->reconstruct_iterations, NULL, _("reconstruction iterations"));
  gtk_box_pack_start(GTK_BOX(page4), g->reconstruct_iterations, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->reconstruct_iterations), "value-changed", G_CALLBACK(reconstruct_iterations_callback), self);

  g->sharpness_feathering = dt_bauhaus_slider_new_with_range(self, 0.1, 3., 0.2, 2., 2);
  dt_bauhaus_slider_set_format(g->sharpness_feathering, "%.2f dB");
  dt_bauhaus_widget_set_label(g->sharpness_feathering, NULL, _("fine edges sensitivity"));
  gtk_box_pack_start(GTK_BOX(page5), g->sharpness_feathering, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->sharpness_feathering), "value-changed", G_CALLBACK(sharpness_feathering_callback), self);

  g->sharpness_offset = dt_bauhaus_slider_new_with_range(self, 0.1, 3., 0.2, 2., 2);
  dt_bauhaus_slider_set_format(g->sharpness_offset, "%.2f dB");
  dt_bauhaus_widget_set_label(g->sharpness_offset, NULL, _("coarse edges sensitivity"));
  gtk_box_pack_start(GTK_BOX(page5), g->sharpness_offset , FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->sharpness_offset), "value-changed", G_CALLBACK(sharpness_offset_callback), self);

  g->sharpness_width = dt_bauhaus_slider_new_with_range(self, 0.01, 3., 0.2, 0.5, 2);
  dt_bauhaus_slider_set_format(g->sharpness_width, "%.2f dB");
  dt_bauhaus_widget_set_label(g->sharpness_width, NULL, _("noise tolerance"));
  gtk_box_pack_start(GTK_BOX(page5), g->sharpness_width , FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->sharpness_width), "value-changed", G_CALLBACK(sharpness_width_callback), self);

  g->dof_sensitivity = dt_bauhaus_slider_new_with_range(self, 0., 200., 0.2, 100., 3);
  dt_bauhaus_slider_set_format(g->dof_sensitivity, "%.2f %%");
  dt_bauhaus_widget_set_label(g->dof_sensitivity, NULL, _("rescale depth of field"));
  gtk_box_pack_start(GTK_BOX(page5), g->dof_sensitivity, FALSE, FALSE, 0);
  g_signal_connect(G_OBJECT(g->dof_sensitivity), "value-changed", G_CALLBACK(dof_sensitivity_callback), self);

  g->sharpness_strength = dt_bauhaus_slider_new_with_range(self, 0., 200., 1., 50., 2);
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
