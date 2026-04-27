from datasets import load_dataset
ds = load_dataset('KoelLabs/L2Arctic', split='scripted', trust_remote_code=True)
ex = ds[0]
print('Columns:', list(ex.keys()))
print('Sample:', {k: v for k, v in ex.items() if k != 'audio'})
