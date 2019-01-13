

def read_configure(name=None, **kwargs):
    import yaml
    import os
    with(open(os.path.join(os.path.expanduser('~'), "collider/config_bundle.yml"))) as f:
        configure = yaml.load(f)

    if name is None and 'name' in kwargs:
        name = kwargs['name']

    if name is None:
        if len(configure) == 1:
            for k, v in configure.items():
                return v
        else:
            raise Exception("Input profile name")
    elif name in configure:
        return configure[name]
    else:
        raise KeyError(name)
