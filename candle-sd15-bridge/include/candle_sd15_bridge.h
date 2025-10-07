#ifndef CANDLE_SD15_BRIDGE_H
#define CANDLE_SD15_BRIDGE_H

// Generated with cbindgen.

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum candle_sd15CandleSd15StatusCode {
  Ok = 0,
  InvalidArgument = 1,
  NotInitialized = 2,
  AlreadyInitialized = 3,
  RuntimeError = 4,
  AssetDirInvalid = 5,
  AssetRootMissing = 6,
  AssetTokenizerMissing = 7,
  AssetClipMissing = 8,
  AssetUnetMissing = 9,
  AssetVaeMissing = 10,
  LoadTokenizerFailed = 11,
  LoadClipFailed = 12,
  LoadUnetFailed = 13,
  LoadVaeFailed = 14,
  MetalUnavailable = 15,
} candle_sd15CandleSd15StatusCode;

typedef struct candle_sd15CandleSd15InitOptions {
  const char *asset_dir;
  uint8_t use_metal;
} candle_sd15CandleSd15InitOptions;

typedef struct candle_sd15CandleSd15Request {
  const char *prompt;
  const char *negative_prompt;
  uint32_t steps;
  uint64_t seed;
  uint8_t use_seed;
} candle_sd15CandleSd15Request;

typedef struct candle_sd15CandleSd15Image {
  uint8_t *data;
  size_t len;
  size_t capacity;
  uint32_t width;
  uint32_t height;
  uint32_t channels;
} candle_sd15CandleSd15Image;

enum candle_sd15CandleSd15StatusCode candle_sd15_init(const struct candle_sd15CandleSd15InitOptions *options);

bool candle_sd15_is_ready(void);

enum candle_sd15CandleSd15StatusCode candle_sd15_generate(const struct candle_sd15CandleSd15Request *request,
                                                          struct candle_sd15CandleSd15Image *out_image);

void candle_sd15_free_image(struct candle_sd15CandleSd15Image *image);

const char *candle_sd15_last_error(void);

#endif /* CANDLE_SD15_BRIDGE_H */
