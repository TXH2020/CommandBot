import yaml
with open('config.yaml','r') as f:
    x=yaml.safe_load(f)
for i in x['hosts']:
    print(tuple(i.values()))