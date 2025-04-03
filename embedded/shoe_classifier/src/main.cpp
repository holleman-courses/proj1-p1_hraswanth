#include <Arduino.h>
#include <TensorFlowLite.h>
#include "model_data.h"
#include "OV767X.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define CAM_WIDTH 224
#define CAM_HEIGHT 224
#define TENSOR_ARENA_SIZE 1024 * 100
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

OV767X Camera;
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setupCamera() {
  if (!Camera.begin(CAM_WIDTH, CAM_HEIGHT, PIXFORMAT_RGB565)) {
    Serial.println("❌ Camera failed");
    while (1);
  }
  Camera.setResolution(CAM_WIDTH, CAM_HEIGHT);
  Camera.setPixelFormat(PIXFORMAT_RGB565);
  delay(100);
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  setupCamera();

  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("❌ Model schema version mismatch!");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("❌ Tensor allocation failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("✅ Setup complete");
}

void loop() {
  Camera.readFrame(input->data.uint8, input->bytes);
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("❌ Inference failed");
    return;
  }

  int score_shoe = output->data.uint8[0];
  int score_nonshoe = output->data.uint8[1];

  Serial.print("Shoe: ");
  Serial.print(score_shoe);
  Serial.print(" | Non-shoe: ");
  Serial.println(score_nonshoe);

  delay(3000);
}