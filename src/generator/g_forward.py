def generator_forward(G, z, y):
    from layers.concat_input import concat_input
    input = concat_input(z, y)
    return G(input)
