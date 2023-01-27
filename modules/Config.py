import yaml

with open('./config.yml', 'r') as f:
    config = yaml.load(f, yaml.Loader)

config['voxelsize'] = [(config['velorange'][i + 3] - config['velorange'][i]) / config['voxelshape'][i] for i in range(3)]

def __getattr__(name):
    return config[name]