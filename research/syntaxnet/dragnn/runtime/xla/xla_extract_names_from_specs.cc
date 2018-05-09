// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// Writes a Bazel file containing a definition for XLA_AOT_COMPONENTS. The
// value is an array; each element is an array of strings containing information
// needed to build the XLA AOT library for a graph, and the DRAGNN component
// that uses it.
//
// This file is loaded and then used by the dragnn_xla_aot_components() build
// rule (see xla_build_defs.bzl). Its contents are verified to be current by the
// dragnn_xla_aot_bazel_test() build rule, which runs this program.
//
// This program processes a set of MasterSpecs; the benefits for processing
// a set of MasterSpecs together are:
//  - only a single build rule is necessary for adding component libraries;
//  - duplicates of model/components across MasterSpecs are flagged as errors.
//
// Usage: xla_extract_names_from_specs graph-base [master-spec-path]+ bazel-path
//   graph-base: base path to remove on GraphDefs in MasterSpecs
//   master-specs: DRAGNN model MasterSpecs (includes base-path)
//   bazel-path: Bazel definition output file

#include <string>
#include <vector>

#include "dragnn/runtime/xla/xla_spec_build_utils.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

int main(int argc, char **argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc < 5) {
    LOG(FATAL)
        << "Usage: xla_extract_names_from_specs"
           "  graph-base [master-spec-path]+ bazel-path\n"
           "  graph-base: base path to remove on GraphDefs in MasterSpecs\n"
           "  master-specs: DRAGNN model MasterSpecs (includes base-path)\n"
           "  bazel-path: Bazel definition output file\n";
  }
  const char *base_path = argv[1];
  std::vector<string> master_spec_paths;
  for (int i = 2; i < argc - 1; i++) {
    master_spec_paths.push_back(argv[i]);
  }
  const string &bazel_path = argv[argc - 1];

  string bazel_def;
  tensorflow::strings::StrAppend(
      &bazel_def,
      "\"\"\"Generated by xla_extract_names_from_specs. "
      "Do not edit.\"\"\"\n\n");
  TF_CHECK_OK(syntaxnet::dragnn::runtime::MasterSpecsToBazelDef(
      "XLA_AOT_COMPONENTS", base_path, master_spec_paths, &bazel_def));
  TF_CHECK_OK(tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                            bazel_path, bazel_def));
  return 0;
}