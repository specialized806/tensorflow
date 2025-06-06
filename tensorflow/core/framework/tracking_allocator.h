/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_FRAMEWORK_TRACKING_ALLOCATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_TRACKING_ALLOCATOR_H_

#include <unordered_map>

#include "xla/tsl/framework/tracking_allocator.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// NOLINTBEGIN(misc-unused-using-decls)
using tsl::AllocRecord;
using tsl::TrackingAllocator;
// NOLINTEND(misc-unused-using-decls)

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TRACKING_ALLOCATOR_H_
