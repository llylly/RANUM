# Structure of ONNX Models

The root object is called `ModelProto`.

The object has:

- `graph`: most important member

- `opset_import`: ProtoBuf list with one element: version

    Retrieve code: `.opset_import[0].version`

- `producer_name`: string, what package creates it

- `ir_version`: int

- ...

Now consider `graph` member, the object has:

- `initializer`: ProtoBuf list, where each element has:

    - `data_type`: int, seems like 1 stands for float
    
    - `name`: str, usually `xxx:0`
    
    - `raw_data`: bytes
    
    Many of elements here are trainable weights

- `input`: ProtoBuf list, where each element has:

    - `name`: str, usually `xxx:0`
    
    - `type`: TypeProto object
        
        - `tensor_type`: Tensor object
        
            - `elem_type`: int, seems like 1 stands for float
    
            - `shape`: TensorShapeProto
    
                - `dim`: a list of Dimension, where each element has
                
                    - `dim_param`: str, stands for the batch size identifier like `unk__xxx`; if not a batch size field, it is an empty string

                    - `dim_value`: int, if it is batch size, the number is 0
    
    Sometimes the names can be overlapped with those names in `initializer`, and they usually correspond to the placeholder with default value.
    So we can deem them as just initializer variables.

- `output`: ProtoBuf list, where each element has the same format as `input` elements
  
- `node`: ProtoBuf list, where each element has:

    - `input`: RepeatedScalarContainer, can directly transform to list of element, where each element is a name in format `xxx:0`
    
    - `output`: Same as `input` format
    
    - *`attribute`: RepeatedCompositeContainer, where each element is an AttributeProto
    
        - `name`: str, attribute name
    
        - `type`: int, indicating type
    
        - `f`, `floats`, `i`, `ints`, ...: actual data
    
    - *`domain`: str, usually just an empty string
    
    - `name`: str, operator name
    
    - `op_type`: str, operator type

- *`value_info`: ProtoBuf list, after `onnx.shape_inference.infer_shapes(...)` this field will exist, where each element has:

    - `name`: str, variable name
    
    - `type`: TypeProto object, same format as `input/type`, and also preserves the batch size identifier

    After model inference, seems to infer almost all variables but not all. Interesting to explore what are inferred.

----

The datatype mapping for tensor definition can be obtained by `dict([(id, x.name) for id, x in enumerate(onnx.TensorProto.DataType._enum_type.values)])`.

```python
{0: 'UNDEFINED',
 1: 'FLOAT',
 2: 'UINT8',
 3: 'INT8',
 4: 'UINT16',
 5: 'INT16',
 6: 'INT32',
 7: 'INT64',
 8: 'STRING',
 9: 'BOOL',
 10: 'FLOAT16',
 11: 'DOUBLE',
 12: 'UINT32',
 13: 'UINT64',
 14: 'COMPLEX64',
 15: 'COMPLEX128',
 16: 'BFLOAT16'}
```

The datatype mapping for attribute definifion can be obtainted by `dict([(id, name) for id, name in enumerate(onnx.AttributeProto.AttributeType.keys())])`.

```python
{0: 'UNDEFINED',
 1: 'FLOAT',
 2: 'INT',
 3: 'STRING',
 4: 'TENSOR',
 5: 'GRAPH',
 6: 'SPARSE_TENSOR',
 7: 'FLOATS',
 8: 'INTS',
 9: 'STRINGS',
 10: 'TENSORS',
 11: 'GRAPHS',
 12: 'SPARSE_TENSORS'}
```
