"""Tests for Ray Data logical operator schema inference."""

import pytest
import pyarrow as pa
from typing import Optional

from ray.data._internal.logical.operators.map_operator import (
    MapRows, Filter, Project, MapBatches, FlatMap
)
from ray.data._internal.logical.operators.all_to_all_operator import (
    StreamingRepartition, Count, Aggregate
)
from ray.data._internal.logical.operators.n_ary_operator import Union
from ray.data._internal.logical.operators.join_operator import Join
from ray.data.block import Block
from ray.data.datasource import Datasource


class MockDatasource(Datasource):
    """Mock datasource for testing."""
    
    def __init__(self, schema: pa.Schema):
        self._schema = schema
    
    def get_read_tasks(self, parallelism: int):
        return []
    
    def estimate_inmemory_data_size(self) -> Optional[int]:
        return None


class TestProjectSchemaInference:
    """Tests for Project operator schema inference."""
    
    def test_project_simple_fields(self):
        """Test projecting simple fields."""
        # 创建输入schema
        input_schema = pa.schema([
            ("id", pa.int64()),
            ("name", pa.string()),
            ("age", pa.int32())
        ])
        
        # 创建mock输入operator
        mock_input = MockOperator(input_schema)
        
        # 创建Project operator
        from ray.data._internal.logical.expressions import ColumnExpr, AliasExpr
        
        # 测试简单列投影
        exprs = [
            ColumnExpr("id"),
            ColumnExpr("name")
        ]
        project = Project(mock_input, exprs)
        
        result_schema = project.infer_schema()
        assert result_schema is not None
        assert len(result_schema) == 2
        assert result_schema.field("id").type == pa.int64()
        assert result_schema.field("name").type == pa.string()
    
    def test_project_with_alias(self):
        """Test projecting with aliases."""
        input_schema = pa.schema([
            ("id", pa.int64()),
            ("name", pa.string())
        ])
        
        mock_input = MockOperator(input_schema)
        
        from ray.data._internal.logical.expressions import ColumnExpr, AliasExpr
        
        # 测试带别名的投影
        exprs = [
            AliasExpr(ColumnExpr("id"), "user_id"),
            AliasExpr(ColumnExpr("name"), "user_name")
        ]
        project = Project(mock_input, exprs)
        
        result_schema = project.infer_schema()
        assert result_schema is not None
        assert len(result_schema) == 2
        assert result_schema.field("user_id").type == pa.int64()
        assert result_schema.field("user_name").type == pa.string()
    
    def test_project_with_star(self):
        """Test projecting with star expression."""
        input_schema = pa.schema([
            ("id", pa.int64()),
            ("name", pa.string()),
            ("age", pa.int32())
        ])
        
        mock_input = MockOperator(input_schema)
        
        from ray.data._internal.logical.expressions import StarExpr, ColumnExpr
        
        # 测试星号表达式
        exprs = [StarExpr()]
        project = Project(mock_input, exprs)
        
        result_schema = project.infer_schema()
        assert result_schema is not None
        assert len(result_schema) == 3
        assert result_schema.field("id").type == pa.int64()
        assert result_schema.field("name").type == pa.string()
        assert result_schema.field("age").type == pa.int32()
    
    def test_project_mixed_expressions(self):
        """Test projecting with mixed expressions."""
        input_schema = pa.schema([
            ("id", pa.int64()),
            ("price", pa.float64()),
            ("quantity", pa.int32())
        ])
        
        mock_input = MockOperator(input_schema)
        
        from ray.data._internal.logical.expressions import (
            ColumnExpr, BinaryExpr, LiteralExpr
        )
        
        # 测试混合表达式
        exprs = [
            ColumnExpr("id"),
            BinaryExpr(ColumnExpr("price"), "*", ColumnExpr("quantity"), "total")
        ]
        project = Project(mock_input, exprs)
        
        result_schema = project.infer_schema()
        assert result_schema is not None
        assert len(result_schema) == 2
        assert result_schema.field("id").type == pa.int64()
        # 二元运算返回float64
        assert result_schema.field("total").type == pa.float64()
    
    def test_project_with_literal(self):
        """Test projecting with literal values."""
        input_schema = pa.schema([
            ("id", pa.int64()),
            ("name", pa.string())
        ])
        
        mock_input = MockOperator(input_schema)
        
        from ray.data._internal.logical.expressions import LiteralExpr
        
        # 测试字面量表达式
        exprs = [
            LiteralExpr(42, "constant_int"),
            LiteralExpr("hello", "constant_str"),
            LiteralExpr(True, "constant_bool")
        ]
        project = Project(mock_input, exprs)
        
        result_schema = project.infer_schema()
        assert result_schema is not None
        assert len(result_schema) == 3
        assert result_schema.field("constant_int").type == pa.int64()
        assert result_schema.field("constant_str").type == pa.string()
        assert result_schema.field("constant_bool").type == pa.bool_()
    
    def test_project_empty_input(self):
        """Test projecting with empty input schema."""
        input_schema = pa.schema([])
        mock_input = MockOperator(input_schema)
        
        from ray.data._internal.logical.expressions import ColumnExpr
        
        exprs = [ColumnExpr("nonexistent")]
        project = Project(mock_input, exprs)
        
        result_schema = project.infer_schema()
        # 应该返回None或处理错误
        assert result_schema is None or len(result_schema) == 1
    
    def test_project_none_input(self):
        """Test projecting when input schema is None."""
        mock_input = MockOperator(None)
        
        from ray.data._internal.logical.expressions import ColumnExpr
        
        exprs = [ColumnExpr("test")]
        project = Project(mock_input, exprs)
        
        result_schema = project.infer_schema()
        assert result_schema is None
    
    def test_project_binary_expressions(self):
        """Test projecting with various binary expressions."""
        input_schema = pa.schema([
            ("a", pa.int64()),
            ("b", pa.int64()),
            ("price", pa.float64()),
            ("name", pa.string())
        ])
        
        mock_input = MockOperator(input_schema)
        
        from ray.data._internal.logical.expressions import BinaryExpr, ColumnExpr
        
        # 测试加法运算
        exprs = [
            BinaryExpr(ColumnExpr("a"), "add", ColumnExpr("b"), "sum_ab"),
            BinaryExpr(ColumnExpr("price"), "mul", ColumnExpr("a"), "total_price"),
            BinaryExpr(ColumnExpr("a"), "eq", ColumnExpr("b"), "is_equal")
        ]
        project = Project(mock_input, exprs)
        
        result_schema = project.infer_schema()
        assert result_schema is not None
        assert len(result_schema) == 3
        assert result_schema.field("sum_ab").type == pa.int64()
        assert result_schema.field("total_price").type == pa.float64()
        assert result_schema.field("is_equal").type == pa.bool_()
    
    def test_project_unary_expressions(self):
        """Test projecting with unary expressions."""
        input_schema = pa.schema([
            ("flag", pa.bool()),
            ("value", pa.int64())
        ])
        
        mock_input = MockOperator(input_schema)
        
        from ray.data._internal.logical.expressions import UnaryExpr, ColumnExpr
        
        # 测试一元表达式
        exprs = [
            UnaryExpr("not", ColumnExpr("flag"), "not_flag"),
            UnaryExpr("neg", ColumnExpr("value"), "neg_value")
        ]
        project = Project(mock_input, exprs)
        
        result_schema = project.infer_schema()
        assert result_schema is not None
        assert len(result_schema) == 2
        assert result_schema.field("not_flag").type == pa.bool_()
        assert result_schema.field("neg_value").type == pa.int64()
    
    def test_project_download_expressions(self):
        """Test projecting with download expressions."""
        input_schema = pa.schema([
            ("url", pa.string()),
            ("filename", pa.string())
        ])
        
        mock_input = MockOperator(input_schema)
        
        from ray.data._internal.logical.expressions import DownloadExpr, ColumnExpr
        
        # 测试下载表达式
        exprs = [
            DownloadExpr(ColumnExpr("url"), "downloaded_content"),
            ColumnExpr("filename")
        ]
        project = Project(mock_input, exprs)
        
        result_schema = project.infer_schema()
        assert result_schema is not None
        assert len(result_schema) == 2
        assert result_schema.field("downloaded_content").type == pa.binary()
        assert result_schema.field("filename").type == pa.string()
    
    def test_project_complex_nested_expressions(self):
        """Test projecting with complex nested expressions."""
        input_schema = pa.schema([
            ("price", pa.float64()),
            ("quantity", pa.int64()),
            ("discount", pa.float64())
        ])
        
        mock_input = MockOperator(input_schema)
        
        from ray.data._internal.logical.expressions import (
            BinaryExpr, ColumnExpr, LiteralExpr
        )
        
        # 测试复杂嵌套表达式
        exprs = [
            BinaryExpr(
                BinaryExpr(ColumnExpr("price"), "mul", ColumnExpr("quantity"), "subtotal"),
                "mul",
                BinaryExpr(LiteralExpr(0.9, None), "sub", ColumnExpr("discount"), None),
                "final_price"
            )
        ]
        project = Project(mock_input, exprs)
        
        result_schema = project.infer_schema()
        assert result_schema is not None
        assert len(result_schema) == 1
        assert result_schema.field("final_price").type == pa.float64()


class MockOperator:
    """Mock operator for testing."""
    
    def __init__(self, schema: Optional[pa.Schema]):
        self._schema = schema
    
    def infer_schema(self) -> Optional[pa.Schema]:
        return self._schema


class TestSimpleOperators:
    """Tests for simple pass-through operators."""
    
    def test_map_rows_schema(self):
        """Test MapRows schema inference."""
        input_schema = pa.schema([
            ("id", pa.int64()),
            ("name", pa.string())
        ])
        mock_input = MockOperator(input_schema)
        
        map_op = MapRows(mock_input, lambda x: x)
        result_schema = map_op.infer_schema()
        
        assert result_schema is not None
        assert result_schema == input_schema
    
    def test_filter_schema(self):
        """Test Filter schema inference."""
        input_schema = pa.schema([
            ("id", pa.int64()),
            ("name", pa.string())
        ])
        mock_input = MockOperator(input_schema)
        
        filter_op = Filter(mock_input, lambda x: True)
        result_schema = filter_op.infer_schema()
        
        assert result_schema is not None
        assert result_schema == input_schema
    
    def test_streaming_repartition_schema(self):
        """Test StreamingRepartition schema inference."""
        input_schema = pa.schema([
            ("id", pa.int64()),
            ("name", pa.string())
        ])
        mock_input = MockOperator(input_schema)
        
        repart_op = StreamingRepartition(mock_input, num_outputs=2)
        result_schema = repart_op.infer_schema()
        
        assert result_schema is not None
        assert result_schema == input_schema
    
    def test_count_schema(self):
        """Test Count schema inference."""
        input_schema = pa.schema([
            ("id", pa.int64()),
            ("name", pa.string())
        ])
        mock_input = MockOperator(input_schema)
        
        count_op = Count(mock_input)
        result_schema = count_op.infer_schema()
        
        assert result_schema is not None
        assert len(result_schema) == 1
        assert result_schema.field("count").type == pa.int64()
    
    def test_union_schema(self):
        """Test Union schema inference."""
        schema1 = pa.schema([
            ("id", pa.int64()),
            ("name", pa.string())
        ])
        schema2 = pa.schema([
            ("id", pa.int64()),
            ("name", pa.string())
        ])
        
        mock_input1 = MockOperator(schema1)
        mock_input2 = MockOperator(schema2)
        
        union_op = Union([mock_input1, mock_input2])
        result_schema = union_op.infer_schema()
        
        assert result_schema is not None
        assert len(result_schema) == 2
        assert result_schema.field("id").type == pa.int64()
        assert result_schema.field("name").type == pa.string()
    
    def test_union_incompatible_schema(self):
        """Test Union with incompatible schemas."""
        schema1 = pa.schema([
            ("id", pa.int64()),
            ("name", pa.string())
        ])
        schema2 = pa.schema([
            ("id", pa.string()),  # 类型不匹配
            ("name", pa.string())
        ])
        
        mock_input1 = MockOperator(schema1)
        mock_input2 = MockOperator(schema2)
        
        union_op = Union([mock_input1, mock_input2])
        result_schema = union_op.infer_schema()
        
        # 应该返回None或处理不兼容的schema
        assert result_schema is None


if __name__ == "__main__":
    pytest.main([__file__])