decimation_factor = 20 # shrink the data decimation_factor*decimation_factor times

field_id_path = '/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldid.npy'

field_id = np.load(field_id_path) # (h, w)
field_type = np.load(field_type_path) # (h, w)

# shrink the data decimation_factor*decimation_factor times
field_id = field_id[::decimation_factor, ::decimation_factor]
field_type = field_type[::decimation_factor, ::decimation_factor]

# save the downsampled field_id and field_type
np.save(f'/maps/zf281/btfm-training-10.4/maddy_code/data_processed/fieldid_{1/decimation_factor**2}.npy', field_id)