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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declare here, so we don't need a public header.
absl::Status ObfuscateNames(const GraphDef& input_graph_def,
                            const TransformFuncContext& context,
                            GraphDef* output_graph_def);

class ObfuscateNamesTest : public ::testing::Test {
 protected:
  void TestSimpleTree() {
    GraphDef graph_def;

    NodeDef* add_node1 = graph_def.add_node();
    add_node1->set_name("add_node1");
    add_node1->set_op("Add");
    add_node1->add_input("add_node2");
    add_node1->add_input("add_node3");

    NodeDef* add_node2 = graph_def.add_node();
    add_node2->set_name("add_node2");
    add_node2->set_op("Add");
    add_node2->add_input("const_node1");
    add_node2->add_input("const_node2");

    NodeDef* add_node3 = graph_def.add_node();
    add_node3->set_name("add_node3");
    add_node3->set_op("Add");
    add_node3->add_input("const_node3");
    add_node3->add_input("const_node4");

    NodeDef* const_node1 = graph_def.add_node();
    const_node1->set_name("const_node1");
    const_node1->set_op("Const");

    NodeDef* const_node2 = graph_def.add_node();
    const_node2->set_name("const_node2");
    const_node2->set_op("Const");

    NodeDef* const_node3 = graph_def.add_node();
    const_node3->set_name("const_node3");
    const_node3->set_op("Const");

    NodeDef* const_node4 = graph_def.add_node();
    const_node4->set_name("const_node4");
    const_node4->set_op("Const");

    GraphDef result;
    TF_ASSERT_OK(
        ObfuscateNames(graph_def, {{"const_node1"}, {"add_node1"}}, &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);

    EXPECT_EQ(1, node_lookup.count("add_node1"));
    EXPECT_EQ(0, node_lookup.count("add_node2"));
    EXPECT_EQ(0, node_lookup.count("add_node3"));
    EXPECT_EQ(1, node_lookup.count("const_node1"));
    EXPECT_EQ(0, node_lookup.count("const_node2"));
    EXPECT_EQ(0, node_lookup.count("const_node3"));
    EXPECT_EQ(0, node_lookup.count("const_node4"));
  }

  void TestManyNodes() {
    GraphDef graph_def;
    for (int i = 0; i < 1000; ++i) {
      NodeDef* const_node = graph_def.add_node();
      const_node->set_name(strings::StrCat("const_node", i));
      const_node->set_op("Const");
    }

    GraphDef result;
    TF_ASSERT_OK(ObfuscateNames(graph_def, {{"const_node0"}, {"const_node999"}},
                                &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("const_node0"));
    EXPECT_EQ(0, node_lookup.count("const_node500"));
    EXPECT_EQ(1, node_lookup.count("const_node999"));
  }

  void TestNameClashes() {
    GraphDef graph_def;
    for (int i = 0; i < 1000; ++i) {
      NodeDef* const_node = graph_def.add_node();
      const_node->set_name(strings::StrCat("1", i));
      const_node->set_op("Const");
    }

    GraphDef result;
    TF_ASSERT_OK(ObfuscateNames(graph_def, {{"10"}, {"19"}}, &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("10"));
    EXPECT_EQ(1, node_lookup.count("19"));

    std::unordered_set<string> names;
    for (const NodeDef& node : result.node()) {
      EXPECT_EQ(0, names.count(node.name()))
          << "Found multiple nodes with name '" << node.name() << "'";
      names.insert(node.name());
    }
  }
};

TEST_F(ObfuscateNamesTest, TestSimpleTree) { TestSimpleTree(); }

TEST_F(ObfuscateNamesTest, TestManyNodes) { TestManyNodes(); }

TEST_F(ObfuscateNamesTest, TestNameClashes) { TestNameClashes(); }

}  // namespace graph_transforms
}  // namespace tensorflow
