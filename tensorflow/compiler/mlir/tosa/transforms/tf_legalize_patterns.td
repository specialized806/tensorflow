/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// TensorFlow legalization patterns

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/Tosa/IR/TosaOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

// Nullary ops patterns.

def : Pat<(TF_ConstOp ElementsAttr : $value), (Tosa_ConstOp $value)>;

// Unary ops patterns.

def : Pat<(TF_IdentityOp $value), (replaceWithValue $value)>;
def : Pat<(TF_AbsOp $arg), (Tosa_AbsOp $arg)>;
def : Pat<(TF_CeilOp $arg), (Tosa_CeilOp $arg)>;
def : Pat<(TF_FloorOp $arg), (Tosa_FloorOp $arg)>;
def : Pat<(TF_ExpOp $arg), (Tosa_ExpOp $arg)>;
def : Pat<(TF_ErfOp $arg), (Tosa_ErfOp $arg)>;
def : Pat<(TF_LogOp $arg), (Tosa_LogOp $arg)>;
def : Pat<(TF_ReciprocalOp $arg), (Tosa_ReciprocalOp $arg)>;
def : Pat<(TF_RsqrtOp $arg), (Tosa_RsqrtOp $arg)>;
def : Pat<(TF_LogicalNotOp $arg), (Tosa_LogicalNotOp $arg)>;
def : Pat<(TF_InvertOp $arg1), (Tosa_BitwiseNotOp $arg1)>;
def : Pat<(TF_CastOp $in, BoolAttr : $truncate), (Tosa_CastOp $in)>;

// Binary ops patterns.
