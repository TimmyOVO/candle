#ifndef CANDLE_SDXL_BRIDGE_H
#define CANDLE_SDXL_BRIDGE_H

// Generated with cbindgen.

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum candle_sdxlCandleSdxlStatusCode {
  Ok = 0,
  InvalidArgument = 1,
  NotInitialized = 2,
  AlreadyInitialized = 3,
  RuntimeError = 4,
  AssetDirInvalid = 5,
  AssetRootMissing = 6,
  AssetTokenizerPrimaryMissing = 7,
  AssetTokenizerSecondaryMissing = 8,
  AssetClipPrimaryMissing = 9,
  AssetClipSecondaryMissing = 10,
  AssetUnetMissing = 11,
  AssetVaeMissing = 12,
  LoadTokenizerPrimaryFailed = 13,
  LoadTokenizerSecondaryFailed = 14,
  LoadClipPrimaryFailed = 15,
  LoadClipSecondaryFailed = 16,
  LoadUnetFailed = 17,
  LoadVaeFailed = 18,
  MetalUnavailable = 19,
} candle_sdxlCandleSdxlStatusCode;

typedef struct candle_sdxlCandleSdxlInitOptions {
  const char *asset_dir;
  uint8_t use_metal;
} candle_sdxlCandleSdxlInitOptions;

typedef struct candle_sdxlCandleSdxlRequest {
  const char *prompt;
  uint32_t steps;
  uint64_t seed;
  uint8_t use_seed;
} candle_sdxlCandleSdxlRequest;

typedef struct candle_sdxlCandleSdxlImage {
  uint8_t *data;
  size_t len;
  size_t capacity;
  uint32_t width;
  uint32_t height;
  uint32_t channels;
} candle_sdxlCandleSdxlImage;

enum candle_sdxlCandleSdxlStatusCode candle_sdxl_init(const struct candle_sdxlCandleSdxlInitOptions *options);

bool candle_sdxl_is_ready(void);

enum candle_sdxlCandleSdxlStatusCode candle_sdxl_generate(const struct candle_sdxlCandleSdxlRequest *request,
                                                          struct candle_sdxlCandleSdxlImage *out_image);

void candle_sdxl_free_image(struct candle_sdxlCandleSdxlImage *image);

const char *candle_sdxl_last_error(void);

#endif /* CANDLE_SDXL_BRIDGE_H */
