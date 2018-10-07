def parse_cfg(cfg_file):
    """
    :param cfg_file: configuration filename
    :return: a list of blocks as dictionary. Each block describes a block in the neutral network.
    """
    with open(cfg_file, 'r') as f:
        lines = f.read().split('\n')
        lines = [x for x in lines if len(x) > 0]
        lines = [x.strip() for x in lines]
        lines = [x for x in lines if x[0] != '#']

        block = {}
        blocks = []

        for line in lines:
            if line[0] == '[':
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
                block['type'] = line[1:-1].strip()
            else:
                key, value = line.split('=')
                block[key.strip()] = value.strip()

        blocks.append(block)
        return blocks
