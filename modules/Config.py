import yaml

with open('./config.yml', 'r') as f:
    config = yaml.load(f, yaml.Loader)

config['voxelsize'] = [(config['velorange'][i + 3] - config['velorange'][i]) / config['voxelshape'][i] for i in range(3)]
if config['half']:
    config['eps'] = 1e-3
else:
    config['eps'] = 1e-6

def __getattr__(name):
    return config[name]