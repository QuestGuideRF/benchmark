// Авторские права (c) Авторы PaddlePaddle, 2018 г. Все права защищены.
//
// Лицензия Apache версии 2.0 («Лицензия»);
// вы не можете использовать этот файл, кроме как в соответствии с Лицензией.
// Вы можете получить копию Лицензии по адресу:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Если это не требуется действующим законодательством или не согласовано в письменной форме, программное обеспечение
// распространяется по Лицензии, распространяется на условиях «КАК ЕСТЬ»,
// БЕЗ КАКИХ-ЛИБО ГАРАНТИЙ ИЛИ УСЛОВИЙ, явных или подразумеваемых.
// См. Лицензию для конкретного языка, регулирующего разрешения и
// ограничения по Лицензии.

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <paddle_inference_api.h>

#ifdef ENABLE_PADDLE_PROFILER
#include <paddle/fluid/platform/profiler.h>
DECLARE_bool(profile);
#endif

DEFINE_string(model_dir, "", "каталог моделей");
DEFINE_string(data, "", "путь входных данных");
DEFINE_int32(repeat, 1, "repeat");
DEFINE_int32(warmup_steps, 0, "repeat");
DEFINE_int32(num_threads, 1, "num_threads");
DEFINE_bool(print_outputs, false, "Выводить ли результаты прогнозирования.");
DEFINE_bool(use_gpu, false, "Используйте ли графический процессор для вывода.");
DEFINE_bool(use_analysis, false, "Используете ли Paddle AnalysisPredictor.");
DEFINE_bool(enable_int8_qat, false, "Включить ли квантовое прогнозирование обучения INT8");
DEFINE_string(ops_to_quantize, "", "Список операций, разделенных запятыми, которые необходимо квантовать");

template <typename T>
void Print(std::string key, T value) {
  std::cout.flags(std::ios::left);
  std::cout << std::setw(20) << key << ": " << value << std::endl;
}

template <>
void Print<bool>(std::string key, bool value) {
  std::cout.flags(std::ios::left);
  if (value) {
    std::cout << std::setw(20) << key << ": истинный" << std::endl;
  } else {
    std::cout << std::setw(20) << key << ": ЛОЖЬ" << std::endl;
  }
}

std::unordered_set<std::string> GetOpsToQuantize() {
  std::unordered_set<std::string> ops_to_quantize;
  std::stringstream ss(FLAGS_ops_to_quantize);

  while (ss.good()) {
    std::string substr;
    getline(ss, substr, ',');
    std::cout << substr << "\n";
    if (!substr.empty())
      ops_to_quantize.insert(substr);
  }

  return ops_to_quantize;
}

void InitFLAGS(int argc, char *argv[]) {
  google::InitGoogleLogging(*argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_dir.empty()) {
    LOG(FATAL) << "пожалуйста, установите каталог модели";
  }

  Print<std::string>("model_dir", FLAGS_model_dir);
  Print<std::string>("data", FLAGS_data);
  Print<int>("repeat", FLAGS_repeat);
  Print<int>("num_threads", FLAGS_num_threads);
  Print<int>("warmup_steps", FLAGS_warmup_steps);
  Print<bool>("print_outputs", FLAGS_print_outputs);
  Print<bool>("use_gpu", FLAGS_use_gpu);
  Print<bool>("use_analysis", FLAGS_use_analysis);
#ifdef ENABLE_PADDLE_PROFILER
  Print<bool>("profile", FLAGS_profile);
#endif
}

template <typename T>
void GetValueFromStream(std::stringstream *ss, T *t) {
  (*ss) >> (*t);
}

template <>
void GetValueFromStream<std::string>(std::stringstream *ss, std::string *t) {
  *t = ss->str();
}

// Разделить строку на вектор
template <typename T>
void Split(const std::string &line, char sep, std::vector<T> *v) {
  std::stringstream ss;
  T t;
  for (auto c : line) {
    if (c != sep) {
      ss << c;
    } else {
      GetValueFromStream<T>(&ss, &t);
      v->push_back(std::move(t));
      ss.str({});
      ss.clear();
    }
  }

  if (!ss.str().empty()) {
    GetValueFromStream<T>(&ss, &t);
    v->push_back(std::move(t));
    ss.str({});
    ss.clear();
  }
}

template <typename T>
constexpr paddle::PaddleDType GetPaddleDType();

template <>
constexpr paddle::PaddleDType GetPaddleDType<int64_t>() {
  return paddle::PaddleDType::INT64;
}

template <>
constexpr paddle::PaddleDType GetPaddleDType<float>() {
  return paddle::PaddleDType::FLOAT32;
}

// Разобрать тензор из строки
template <typename T>
bool ParseTensor(const std::string &field, paddle::PaddleTensor *tensor) {
  std::vector<std::string> data;
  Split(field, ':', &data);
  if (data.size() < 2) return false;

  std::string shape_str = data[0];

  std::vector<int> shape;
  Split(shape_str, ' ', &shape);

  std::string mat_str = data[1];

  std::vector<T> mat;
  Split(mat_str, ' ', &mat);

  tensor->shape = shape;
  auto size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
      sizeof(T);
  tensor->data.Resize(size);
  std::copy(mat.begin(), mat.end(), static_cast<T *>(tensor->data.data()));
  tensor->dtype = GetPaddleDType<T>();

  return true;
}

// Разобрать входные тензоры из строки
bool ParseLine(const std::string &line,
               std::vector<paddle::PaddleTensor> *tensors,
               int inputs_num) {
  std::vector<std::string> fields;
  Split(line, ';', &fields);

  // if (fields.size() != 7) return false;

  tensors->clear();
  tensors->reserve(inputs_num);

  int i = 0;
  // src_ids
  paddle::PaddleTensor src_ids;
  ParseTensor<int64_t>(fields[i++], &src_ids);
  src_ids.name = "placeholder_0";
  tensors->push_back(src_ids);

  // pos_ids
  paddle::PaddleTensor pos_ids;
  ParseTensor<int64_t>(fields[i++], &pos_ids);
  pos_ids.name = "placeholder_1";
  tensors->push_back(pos_ids);

  if (inputs_num > 2) {
    // sent_ids
    paddle::PaddleTensor sent_ids;
    ParseTensor<int64_t>(fields[i++], &sent_ids);
    sent_ids.name = "placeholder_2";
    tensors->push_back(sent_ids);

    // input_mask
    paddle::PaddleTensor input_mask;
    ParseTensor<float>(fields[i++], &input_mask);
    input_mask.name = "placeholder_3";
    tensors->push_back(input_mask);
  }

  return true;
}

static int AssessInputNum(std::ifstream& file) {
  // Оцените количество входов на основе количества двоеточий
  // присутствует в первой строке входных данных.
  std::string line;
  std::getline(file, line);
  file.seekg (0, std::ios::beg);

  int colons_num = std::count(line.begin(), line.end(), ':');
  int inputs_num = colons_num == 2 ? colons_num : 4;

  return inputs_num;
}

bool LoadInputData(std::vector<std::vector<paddle::PaddleTensor>> *inputs) {
  if (FLAGS_data.empty()) {
    LOG(ERROR) << "пожалуйста, установите путь к входным данным";
    return false;
  }

  std::ifstream fin(FLAGS_data);
  std::string line;

  const int inputs_num = AssessInputNum(fin);

  int lineno = 0;
  while (std::getline(fin, line)) {
    std::vector<paddle::PaddleTensor> feed_data;
    if (!ParseLine(line, &feed_data, inputs_num)) {
      LOG(ERROR) << "Разобрать строку[" << lineno << "] ошибка!";
    } else {
      inputs->push_back(std::move(feed_data));
    }
    lineno++;
  }

  LOG(INFO) << "Load " << lineno << " образцы из " << FLAGS_data;
  return true;
}

void PrintOutputs(const std::vector<paddle::PaddleTensor> &outputs, int id, double time) {
  //LOG(INFO) << "example_id\tcontradiction\tentailment\tneutral";
  for (size_t i = 0; i < outputs.front().data.length() / sizeof(float) / 3; i += 1) {
    std::cout.flags(std::ios::right);
    std::cout << "example " << std::setw(5) << id
              << ", [" << std::setw(12) << static_cast<float *>(outputs[0].data.data())[3 * i]
              << ", " << std::setw(12) << static_cast<float *>(outputs[0].data.data())[3 * i + 1]
              << ", " << std::setw(12) << static_cast<float *>(outputs[0].data.data())[3 * i + 2]
              << "], время: " << time << " ms" << std::endl;
  }
}

template <typename ConfigType>
void SetConfig(ConfigType* config, std::string model_dir, bool use_gpu, bool use_zerocopy) {
  config->model_dir = model_dir;
  if (use_gpu) {
    config->use_gpu = true;
    config->device = 0;
    config->fraction_of_gpu_memory = 0.15;
  } else {
    config->use_gpu = false;
    config->SetCpuMathLibraryNumThreads(FLAGS_num_threads);
  }
}

template <>
void SetConfig<paddle::AnalysisConfig>(paddle::AnalysisConfig* config, std::string model_dir,
                               bool use_gpu, bool use_zerocopy) {
  config->SetModel(model_dir);
  if (use_gpu) {
    config->EnableUseGpu(100, 0);
  } else {
    config->DisableGpu();
    config->EnableMKLDNN();
    if (FLAGS_enable_int8_qat) {
      std::unordered_set<std::string> ops_to_quantize = GetOpsToQuantize();
      config->EnableMkldnnInt8(ops_to_quantize);
    }
    // поэтому я отключаю константу_folding_pass
    auto pass_builder = config->pass_builder();
    pass_builder->DeletePass("constant_folding_pass");
    config->SetCpuMathLibraryNumThreads(FLAGS_num_threads);
  }
  config->SwitchIrOptim(true);
  // config.SwitchSpecifyInputNames();
  if (use_zerocopy) {
    config->SwitchUseFeedFetchOps(false);
  }
  // конфигурация->SwitchIrDebug();
}

std::unique_ptr<paddle::PaddlePredictor> CreatePredictor(
        const paddle::PaddlePredictor::Config *config, bool use_analysis = true) {
  const auto *analysis_config = reinterpret_cast<const paddle::AnalysisConfig *>(config);
  if (use_analysis) {
    return paddle::CreatePaddlePredictor<paddle::AnalysisConfig>(*analysis_config);
  }
  auto native_config = analysis_config->ToNativeConfig();
  return paddle::CreatePaddlePredictor<paddle::NativeConfig>(native_config);
}

struct Timer {
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(startu -
          start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};

int main(int argc, char *argv[]) {
  InitFLAGS(argc, argv);

  paddle::AnalysisConfig config;
  SetConfig<paddle::AnalysisConfig>(&config, FLAGS_model_dir, FLAGS_use_gpu, false);
  auto predictor = CreatePredictor(
      reinterpret_cast<paddle::PaddlePredictor::Config *>(&config), FLAGS_use_analysis);

  std::vector<std::vector<paddle::PaddleTensor>> inputs;
  if (!LoadInputData(&inputs)) {
    LOG(ERROR) << "ошибка при загрузке входных данных!";
    return -1;
  }

#ifdef ENABLE_PADDLE_PROFILER
  if (FLAGS_profile) {
    if (!FLAGS_use_gpu) {
      paddle::platform::EnableProfiler(paddle::platform::ProfilerState::kCPU);
    } else {
      paddle::platform::EnableProfiler(paddle::platform::ProfilerState::kAll);
    }
  }
#endif

  std::vector<paddle::PaddleTensor> fetch;
  double total_time{0};
  double total_time_without_warmup{0};
  int num_samples{0};
  int num_samples_without_warmup{0};
  for (int repeat = 0; repeat < FLAGS_repeat; repeat++) {
    for (int id = 0; id < inputs.size(); ++id) {
      fetch.clear();

      Timer timer;
      timer.tic();
      predictor->Run(inputs[id], &fetch);
      double runtime = timer.toc();

      if (FLAGS_print_outputs && (repeat == 0)) {
        PrintOutputs(fetch, id, runtime);
      }
      if (!fetch.empty()) {
        total_time += runtime;
        num_samples += fetch.front().data.length() / (sizeof(float) * 3);
        LOG(INFO) << "--- итерация " << id;
        if (!((repeat == 0) && (id < FLAGS_warmup_steps))) {
          total_time_without_warmup += runtime;
          num_samples_without_warmup += fetch.front().data.length() / (sizeof(float) * 3);
        } else {
#ifdef ENABLE_PADDLE_PROFILER
          if (FLAGS_profile) {
            paddle::platform::ResetProfiler();
          }
#endif
        }
      }
    }
  }

#ifdef ENABLE_PADDLE_PROFILER
  if (FLAGS_profile) {
    paddle::platform::DisableProfiler(
        paddle::platform::EventSortingKey::kTotal, "ernie.inference.profile");
  }
#endif

  double per_sample_ms =
      total_time / static_cast<double>(num_samples);
  LOG(INFO) << "Run " << num_samples
            << " семплы, средняя задержка: " << per_sample_ms
            << " мс на выборку.";
  double per_sample_ms_without_warmup =
      total_time_without_warmup / static_cast<double>(num_samples_without_warmup);
  LOG(INFO) << "Run " << num_samples_without_warmup
            << " образцы, средняя задержка [исключить " << FLAGS_warmup_steps << " этапы разминки]: "
            << per_sample_ms_without_warmup << " мс на выборку.";

  return 0;
}
