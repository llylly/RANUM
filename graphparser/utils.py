"""
    Some useful functions on editing ONNX model
"""

def remove_node_and_decendents(onnx_model, node_name):
    graph = onnx_model.graph
    inv_link = dict()
    for node in graph.node:
        for input in node.input:
            if input not in inv_link:
                inv_link[input] = [node.name]
            else:
                inv_link[input].append(node.name)
        for output in node.output:
            if node.name not in inv_link:
                inv_link[node.name] = [output]
            else:
                inv_link[node.name].append(output)

    # name_node_map = dict([(x.name, x) for x in graph.node])

    que = [node_name]
    vis = {node_name}
    l = 0
    while l < len(que):
        if que[l] in inv_link:
            for nex in inv_link[que[l]]:
                if nex not in vis:
                    vis.add(nex)
                    que.append(nex)
        l += 1

    node_to_rm = [node for node in graph.node if node.name in vis]
    initializer_to_rm = [item for item in graph.initializer if item.name in vis]
    input_to_rm = [item for item in graph.input if item.name in vis]
    output_to_rm = [item for item in graph.output if item.name in vis]

    for x in node_to_rm:
        graph.node.remove(x)
    for x in initializer_to_rm:
        graph.initializer.remove(x)
    for x in input_to_rm:
        graph.input.remove(x)
    for x in output_to_rm:
        graph.output.remove(x)

    return onnx_model

