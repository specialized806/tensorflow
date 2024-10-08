# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "hexagon_implementation",
    srcs = ["hexagon_implementation.cc"],
    hdrs = [
        "hexagon_implementation.h",
        "hexagon_nn_interface.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/delegates/hexagon/hexagon_nn:hexagon_nn_header",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_library(
    name = "hexagon_delegate_kernel",
    srcs = [
        "hexagon_delegate.h",
        "hexagon_delegate_kernel.cc",
    ],
    hdrs = ["hexagon_delegate_kernel.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        ":hexagon_implementation",
        ":utils",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/hexagon/builders:op_builder",
        "//tensorflow/lite/delegates/hexagon/hexagon_nn:hexagon_nn_header",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/schema:schema_fbs",
        "@hexagon_nn//:hexagon_nn_header",
        "@hexagon_nn//:hexagon_nn_ops",
    ],
)

cc_library(
    name = "enable_hexagon_delegate",
    defines = select({
        "//tensorflow:arm_any": ["TFLITE_ENABLE_HEXAGON"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "hexagon_delegate",
    srcs = ["hexagon_delegate.cc"],
    hdrs = ["hexagon_delegate.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        ":hexagon_delegate_kernel",
        ":hexagon_implementation",
        ":utils",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/utils:simple_delegate",
    ] + select({
        "//tensorflow:ios": [],
        "//tensorflow:ios_x86_64": [],
        "//tensorflow:macos": [],
        "//conditions:default": [":enable_hexagon_delegate"],
    }),
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    linkopts = tflite_linkopts() + ["-lm"],
    deps = [
        ":utils",
        "//tensorflow/lite/core/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

exports_files([
    "hexagon_delegate.h",
    "version_script.lds",
])
