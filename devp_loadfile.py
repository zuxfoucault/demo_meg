import mne
#raw = mne.io.read_raw_kit('/home/foucault/data/sourcedata/working_memory/SUBJECT_225_MEG_150729/EM.con')
#raw = mne.io.read_raw_kit('/home/foucault/data/sourcedata/working_memory/SUBJECT_225_MEG_150729/Temp010203_001.con')


from mne_bids import write_raw_bids, read_raw_bids, make_bids_basename
fname = "/home/foucault/data/rawdata/working_memory/sub-106/meg/sub-106_task-Spat030102_meg.con"
output_path = "/home/foucault/data/rawdata/working_memory"
bids_basename = make_bids_basename(subject='106', session=None, task='Spat030102', run=None)
#print(bids_basename)
raw = read_raw_bids(bids_basename + '_meg.con', output_path)
events, event_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, event_id)
#raw = read_raw_bids(fname, output_path)


__import__('IPython').embed()

