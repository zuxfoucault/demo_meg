import os
import loggy
import copy
from pathlib import Path
import pickle
import mne
import numpy as np
#import matplotlib.pyplot as plt
import mne_bids
from mne_bids import read_raw_bids, make_bids_basename

PROJECTS_ROOT = "/home/foucault/projects/"
PROJECT_NAME = "aging_multimodality/"

try:
    LOGDIR = os.path.join(PROJECTS_ROOT, PROJECT_NAME, "logs/")
    file_name = LOGDIR+os.path.basename(__file__)+".log"
    logger = loggy.logger(file_name, __name__)
except NameError:
    file_name = "/dev/null"
    log_name = "jupyter"
    print("In Jupyter")
    logger = loggy.logger(file_name, log_name)


#fname = "/home/foucault/data/rawdata/working_memory/sub-106/meg/sub-106_task-Spat030102_meg.con"


class Logger(object):
    def __init__(self):
        self.info = print


def read_csv():
    import csv
    import codecs
    name_d = "/home/foucault/data/sourcedata/eprime_merged_proc/SUBJECT_106/Untitled.csv"
    doc = codecs.open('document','rU','UTF-16')
    with open(name_d, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            print(f'\t{row["name"]} works in the {row["department"]} department, and was born in {row["birthday month"]}.')
            line_count += 1
        print(f'Processed {line_count} lines.')
    __import__('IPython').embed()
    sys.exit()


def head_filter():
    " routines after exporting data from eprime "
    name_d = "/home/foucault/data/sourcedata/eprime_merged_proc"
    p = Path(name_d)
    f_list = list(p.glob('**/Untitled.txt'))
    for i_file in f_list:
        print(f"processing: {i_file}")
        f_write = i_file.parent.joinpath("head_filter.csv")
        head_transform(i_file.as_posix(), f_write.as_posix())


def head_transform(f_read, f_write):
    " subroutine of the transformation "
    import codecs
    f = codecs.open(f_read, 'rU', 'UTF-16')
    fw = open(f_write, 'w')
    data = f.readlines()
    if 'emrg2' in data[0]:
        print(f"find emrg2")
        fw.writelines(data[1:])
    else:
        fw.writelines(data)


def correct_time_head_filter():
    " re-order time; not test yet "
    import pandas as pd
    import codecs
    name_d = "/home/foucault/data/sourcedata/eprime_merged_proc"
    p = Path(name_d)
    f_list = list(p.glob('**/head_filter.csv'))
    for i_file in f_list:
        df = pd.read_csv(i_file, sep='\t')
        # but just do this while reading df at run-time
        # read_eprime_csv
        _df = df.sort_values(by=['SessionTime'])
        f_write = i_file.parent.joinpath("head_filter_tc.csv")
        _df.to_csv(f_write.as_posix())
        __import__('IPython').embed()
        __import__('sys').exit()


def read_lines():
    " devp "
    import codecs
    name_d = "/home/foucault/data/sourcedata/eprime_merged_proc/SUBJECT_106/Untitled.txt"
    name_d = "/home/foucault/data/sourcedata/eprime_merged_proc/SUBJECT_106/Untitled.csv"
    name_dw = "/home/foucault/data/sourcedata/eprime_merged_proc/SUBJECT_106/head_filter.csv"
    f = codecs.open(name_d, 'rU', 'UTF-16')
    fw = open(name_dw, 'w')
    #f=open(doc, "r")
    #f=open(name_d, "r")
    data = f.readlines()
    if 'emrg2' in data[0]:
        print(f"find emrg2")
        fw.writelines(data[1:])
    else:
        fw.writelines(data)
    __import__('IPython').embed()
    __import__('sys').exit()
    for i in f.readlines():
        if 'emrg2' not in i:
            print(f"find emrg2")
        __import__('IPython').embed()


def read_csv2():
    " check head_filter transformation results "
    import pandas as pd
    import codecs
    name_d = "/home/foucault/data/sourcedata/eprime_merged_proc"
    p = Path(name_d)
    f_list = list(p.glob('**/head_filter.csv'))
    for i_file in f_list:
        df = pd.read_csv(i_file, sep='\t')
        __import__('IPython').embed()
        __import__('sys').exit()


def read_csv3():
    " devp "
    import pandas as pd
    import codecs
    name_d = "/home/foucault/data/sourcedata/eprime_merged_proc/SUBJECT_106/Untitled.csv"
    name_d = "/home/foucault/data/sourcedata/eprime_merged_proc/SUBJECT_106/Untitled.txt"
    name_d = "/home/foucault/data/sourcedata/eprime_merged_proc/SUBJECT_106/head_filter.csv"
    # doc = codecs.open(name_d, 'rU', 'UTF-16')
    # df = pd.read_csv(doc, sep='\t')
    df = pd.read_csv(name_d, sep='\t')
    __import__('IPython').embed()
    __import__('sys').exit()


def devp_bids():
    " devp "
    output_path = "/home/foucault/data/rawdata/working_memory"
    bids_basename = make_bids_basename(subject='106', session=None, task='Spat030102', run=None)
    print(bids_basename)
    raw = read_raw_bids(bids_basename + '_meg.con', output_path)
    # events, event_id = mne.events_from_annotations(raw)
    # epochs = mne.Epochs(raw, events, event_id)
    events = mne.find_events(raw, 'STI 014', consecutive=True)
    event_id = list(np.unique(events[:,2]))
    epochs = mne.Epochs(raw, events, event_id)
    return raw, events, event_id


def devp_raw():
    " devp "
    raw_f = '/home/foucault/data/sourcedata/working_memory/SUBJECT_106/MEG/Spat030102.con'
    mrk_f = '/home/foucault/data/sourcedata/working_memory/SUBJECT_106/MEG/150129-1.mrk'
    elp_f = '/home/foucault/data/sourcedata/working_memory/SUBJECT_106/MEG/001.elp'
    hsp_f = '/home/foucault/data/sourcedata/working_memory/SUBJECT_106/MEG/001.hsp'
    raw = mne.io.read_raw_kit(input_fname=raw_f, mrk=mrk_f, elp=elp_f,
    hsp=hsp_f, allow_unknown_format=True)
    # raw_f = '/home/foucault/data/sourcedata/working_memory/SUBJECT_230/MEG/002_Temp020301.con'
    # raw = mne.io.read_raw_kit(input_fname=raw_f)
    # events, event_id = mne.events_from_annotations(raw)
    events = mne.find_events(raw, 'STI 014', consecutive=True)
    event_id = list(np.unique(events[:,2]))
    epochs = mne.Epochs(raw, events, event_id)
    return raw, events, event_id


def example():
    " devp "
    #import mne
    #from mne.datasets import sample
    #print(__doc__)
    #data_path = sample.data_path()
    #fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
    #raw = mne.io.read_raw_fif(fname)
    #__import__('IPython').embed()
    #__import__('sys').exit()

    # Set up pick list: MEG + STI 014 - bad channels
    want_meg = True
    want_eeg = False
    want_stim = False
    include = ['STI 014']
    #raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bad channels + 2 more

    picks = mne.pick_types(raw.info, meg=want_meg, eeg=want_eeg, stim=want_stim,
                        include=include)

    some_picks = picks[:5]  # take 5 first
    start, stop = raw.time_as_index([0, 15])  # read the first 15s of data
    data, times = raw[some_picks, start:(stop + 1)]

    # save 150s of MEG data in FIF file
    # raw.save('sample_audvis_meg_trunc_raw.fif', tmin=0, tmax=150, picks=picks,
            # overwrite=True)


def plot_events_id():
    raw2, events2, event_id2 = devp_bids()
    fig, ax = plt.subplots(figsize=(20, 10))
    __import__('IPython').embed()
    __import__('sys').exit()
    start = 100
    end = 300
    ax.stem(events2[:,0][start:end], events2[:,2][start:end], bottom=220, use_line_collection=True)
    # ax.stem(1e3 * times, data, bottom=200, use_line_collection=True)
    fig.tight_layout()
    #fig = raw.plot(show=False)
    fig.savefig("viz_event_id.png")


def plot_time():
    " devp "
    raw2, events2, event_id2 = devp_bids()
    fig, ax = plt.subplots(figsize=(20, 10))
    start = 100
    end = 300
    raw1, events1, event_id1 = devp_raw()
    raw1, events1, event_id1 = devp_raw()
    __import__('IPython').embed()
    __import__('sys').exit()
    #ax.stem(events2[:,0][start:end], events2[:,2][start:end], bottom=220, use_line_collection=True)
    # ax.stem(1e3 * times, data, bottom=200, use_line_collection=True)
    fig.tight_layout()
    #fig = raw.plot(show=False)
    fig.savefig("viz_event_id.png")


def preprocess_pipeline():
    raw2, events2, event_id2 = devp_bids()
    raw1, events1, event_id1 = devp_raw()
    __import__('IPython').embed()
    __import__('sys').exit()
    #epochs = mne.Epochs(raw2, events2, event_id2)
    epochs = mne.Epochs(raw2, events2, event_id=223, tmin=-0.1, tmax=1, preload=True)
    # Downsample to 100 Hz
    print('Original sampling rate:', epochs.info['sfreq'], 'Hz')
    epochs_resampled = epochs.copy().resample(250, npad='auto')
    print('New sampling rate:', epochs_resampled.info['sfreq'], 'Hz')
    data = epochs_resampled.get_data()
    data_r = epochs.get_data()


def devp_ssp_em():
    #trans = mne_bids.get_head_mri_trans(bids_basename + '_meg.con', root_dir)
    raw = read_raw_bids(bids_basename + '_meg.con', root_dir)
    em = read_raw_bids(bids_basename_em + '_meg.con', root_dir)
    # events, event_id = mne.events_from_annotations(raw)
    # epochs = mne.Epochs(raw, events, event_id)
    # events = mne.find_events(raw, 'STI 014', consecutive=True)
    # event_id = list(np.unique(events[:,2]))
    # epochs = mne.Epochs(raw, events, event_id)
    __import__('IPython').embed()
    __import__('sys').exit()


def define_id_list():
    id_list = [104, 106, 108, 111, 112, 113, 114, 115, 116, 117, 118, 119,
               120, 121, 122, 123, 124, 125, 126, 127, 128, 201, 202, 203,
               204, 205, 206, 208, 209, 210, 212, 214, 215, 216, 217, 218,
               219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230]
    exclude_list = ['108', '116', '119', '120', '122', '202',
                    '203', '206', '210', '223']
    data_concated_dict = {'126': concatenate_raw_sub126,
                          '228': concatenate_raw_sub228}
    # '108' Count by Spat: 42, Count by Temp: 63
    # [Errno 2] No such file or directory: '/home/foucault/data/rawdata/working_memory/sub-116/meg/sub-116_task-EM_meg.con'
    # [Errno 2] No such file or directory: '/home/foucault/data/rawdata/working_memory/sub-119/meg/sub-119_task-EM_meg.con'
    # [Errno 2] No such file or directory: '/home/foucault/data/rawdata/working_memory/sub-122/meg/sub-122_task-EM_meg.con'
    # [Errno 2] No such file or directory: '/home/foucault/data/rawdata/working_memory/sub-202/meg/sub-202_task-EM_meg.con'
    # [Errno 2] No such file or directory: '/home/foucault/data/rawdata/working_memory/sub-203/meg/sub-203_task-EM_meg.con'
    # [Errno 2] No such file or directory: '/home/foucault/data/rawdata/working_memory/sub-206/meg/sub-206_task-EM_meg.con'
    # [Errno 2] No such file or directory: '/home/foucault/data/rawdata/working_memory/sub-210/meg/sub-210_task-EM_meg.con'
    # 223 no temporal task
    # 120 connection of data; data not continuous collected
    # 126 connection of data; not identified sequence
    # 228 connection of data; seq

    # 205 ica.exclude = ica.labels_['blink'], KeyError: 'blink'
    # 209 ica.exclude = ica.labels_['blink'], KeyError: 'blink'
    # 212 ica.exclude = ica.labels_['blink'], KeyError: 'blink'
    # 216 ica.exclude = ica.labels_['blink'], KeyError: 'blink'
    # 217 ica.exclude = ica.labels_['blink'], KeyError: 'blink'
    # 227 ica.exclude = ica.labels_['blink'], KeyError: 'blink'
    # __import__('IPython').embed()
    # __import__('sys').exit()
    return id_list, exclude_list, data_concated_dict


def define_id_by_aging_dict():
    young_list = [104, 106, 108, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                  120, 121, 122, 123, 124, 125, 126, 127, 128]
    elder_list = [201, 202, 203, 204, 205, 206, 208, 209, 210, 212, 214, 215,
                  216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
                  228, 229, 230]
    return {'young': young_list, 'elder': elder_list}


def extract_id_array():
    " read con files; extract sub ids "
    import numpy as np
    root_dir = "/home/foucault/data/rawdata/working_memory"
    p = Path(root_dir)
    f_list = list(p.glob('**/*Spat*_meg.con'))
    id_list = list()
    for i_file in f_list:
        print(f"processing: {i_file}")
        id_list.append(int(i_file.parts[-1][4:7]))
    id_array = np.unique(id_list)
    return id_array


def build_id_task_dict():
    " read con files; extract sub ids "
    import numpy as np
    import pandas as pd
    root_dir = "/home/foucault/data/rawdata/working_memory"
    p = Path(root_dir)
    f_list = list(p.glob('**/*Spat*_meg.con'))
    id_task_dict = dict()
    id_list = list()
    for i_file in f_list:
        print(f"processing: {i_file}")
        id_int = int(i_file.parts[-1][4:7])
        id_list.append(id_int)
        id_task_dict[id_int] = list()
        id_task_dict[id_int].append(i_file.parts[-1][13:].strip('_meg.con'))
        #id_task_dict[id_int] = i_file.parts[-1][13:].strip('_meg.con')

    f_list = list(p.glob('**/*Temp*_meg.con'))
    for i_file in f_list:
        print(f"processing: {i_file}")
        id_int = int(i_file.parts[-1][4:7])
        id_list.append(id_int)
        if id_int not in id_task_dict.keys():
            id_task_dict[id_int] = list()
        id_task_dict[id_int].append(i_file.parts[-1][13:].strip('_meg.con'))
        #id_task_dict[id_int] = i_file.parts[-1][13:].strip('_meg.con')
    id_array = np.unique(id_list)
    root_dir = Path('/home/foucault/data/derivatives/working_memory/intermediate')
    data_output = root_dir.joinpath('id_task_dict.csv')
    df = pd.DataFrame.from_dict(id_task_dict, orient='index')
    df = df.sort_index()
    print("write intermediate id_task_dict:")
    print(f"{data_output}")
    df.to_csv(data_output)
    df = pd.DataFrame(id_array)
    data_output = root_dir.joinpath('id_array.csv')
    print("write intermediate id_array:")
    print(f"{data_output}")
    df.to_csv(data_output)


def read_id_task_dict():
    import pandas as pd
    root_dir = Path('/home/foucault/data/derivatives/working_memory/intermediate')
    data_in = root_dir.joinpath('id_task_dict.csv')
    column = ['sub_id', '0', '1', '2']
    df = pd.read_csv(data_in, names=column, header=0)
    #df = pd.read_csv(data_in)
    return df


def read_id_array():
    import pandas as pd
    root_dir = Path('/home/foucault/data/derivatives/working_memory/intermediate')
    data_in = root_dir.joinpath('id_array.csv')
    df = pd.read_csv(data_in)
    return df


def read_eprime_csv(sub_index=None):
    " check head_filter transformation results "
    import pandas as pd
    df = read_id_task_dict()
    column = ['sub_id', '0', '1', '2']
    if sub_index == None:
        sub_index = 0
    subject_id = str(df[column[0]][sub_index])
    name_d = "/home/foucault/data/sourcedata/eprime_merged_proc"
    p = Path(name_d)
    #sub_id = 104
    i_file = p.glob(f'SUBJECT_{subject_id}/head_filter.csv')
    i_file = next(i_file)
    print(f"{i_file}")
    #i_file = list(p.glob(f'SUBJECT_{subject_id}/head_filter.csv'))
    df = pd.read_csv(i_file, sep='\t')
    # re-order by time
    df = df.sort_values(by=['SessionTime'])
    count = 0
    for i in df['ExperimentName']:
        if 'Spat' in i:
            count += 1
    logger.info(f"Count by Spat: {count}") # n=63
    count = 0
    for i in df['ExperimentName']:
        if 'Temp' in i:
            count += 1
    logger.info(f"Count by Temp: {count}") # n=63
    # __import__('IPython').embed()
    # __import__('sys').exit()
    return df


def concatenate_raw_sub126(task_selected):
    """ for sub-id 18, the sequence is 213 """
    import mne
    if task_selected == 'spatial':
        task=['0012Spat02', '0013Spat0103']
        # look at temporal task sequence
        raw, events, event_id, raw_em = \
            devp_read_raw_bids_by_task(sub_index=18, task=task[0])
        raw_, events_, event_id_, raw_em_ = \
            devp_read_raw_bids_by_task(sub_index=18, task=task[1])
        new_raw = raw.copy()
        new_raw = mne.io.concatenate_raws([new_raw.load_data(), raw_.load_data()])
        new_raw._init_kwargs['input_fname'] = raw._init_kwargs['input_fname'][:70]+'2Spat020103_meg.con'
        events = mne.find_events(new_raw, 'STI 014', consecutive=True, min_duration=1,
                                initial_event=True)
        event_id = list(np.unique(events[:,2]))
    else:
        sub_index = 18
        new_raw, events, event_id, raw_em = _read_raw_bids(sub_index, task_selected)
    return new_raw, events, event_id, raw_em


def concatenate_raw_sub228(task_selected):
    """ for sub-id 45, the sequence is 213 """
    import mne
    if task_selected == 'spatial':
        task=['0012Spat0102', '0014Spat03']
        # look at temporal task sequence
        raw, events, event_id, raw_em = \
            devp_read_raw_bids_by_task(sub_index=45, task=task[0])
        raw_, events_, event_id_, raw_em_ = \
            devp_read_raw_bids_by_task(sub_index=45, task=task[1])
        new_raw = raw.copy()
        new_raw = mne.io.concatenate_raws([new_raw.load_data(), raw_.load_data()])
        new_raw._init_kwargs['input_fname'] = raw._init_kwargs['input_fname'][:70]+'001Spat010203_meg.con'
        events = mne.find_events(new_raw, 'STI 014', consecutive=True, min_duration=1,
                                initial_event=True)
        event_id = list(np.unique(events[:,2]))
    else:
        sub_index = 45
        new_raw, events, event_id, raw_em = _read_raw_bids(sub_index, task_selected)
    return new_raw, events, event_id, raw_em


def devp_read_raw_bids_by_task(sub_index=None, task=None):
    """ for concatenate data """
    df = read_id_task_dict()
    root_dir = "/home/foucault/data/rawdata/working_memory"
    column = ['sub_id', '0', '1', '2']
    subject_id = str(df[column[0]][sub_index])
    if sub_index == None:
        sub_index = 0
    # task_desc_dict = dict(zip(['subject', 'spatial', 'temporal', 'mis'], column))
    # if task_selected == None:
        # task_selected = 'spatial'
        # #task_selected = 'temporal'
    # task = df[task_desc_dict[task_selected]][sub_index]
    # #task = df[column[1]][sub_index]
    bids_basename = make_bids_basename(
        subject=subject_id, session=None, task=task, run=None)
    bids_basename_em = make_bids_basename(
        subject=subject_id, session=None, task='EM', run=None)
    raw_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_meg.con'
    mrk_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_markers.mrk'
    elp_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_elp.elp'
    hsp_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_hsp.hsp'
    raw_em_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename_em+'_meg.con'
    print(bids_basename)
    raw = mne.io.read_raw_kit(input_fname=raw_f, mrk=mrk_f, elp=elp_f,
        hsp=hsp_f, allow_unknown_format=True)
    raw_em = mne.io.read_raw_kit(input_fname=raw_em_f, mrk=mrk_f, elp=elp_f,
        hsp=hsp_f, allow_unknown_format=True)
    events = mne.find_events(raw, 'STI 014', consecutive=True, min_duration=1,
                             initial_event=True)
    #events = mne.find_events(raw, 'STI 014', consecutive=True)
    event_id = list(np.unique(events[:,2]))
    #epochs = mne.Epochs(raw, events, event_id)
    return raw, events, event_id, raw_em
    #return raw


def devp_read_raw_bids(sub_index=None, task_selected=None):
    df = read_id_task_dict()
    root_dir = "/home/foucault/data/rawdata/working_memory"
    column = ['sub_id', '0', '1', '2']
    if sub_index == None:
        sub_index = 0
    task_desc_dict = dict(zip(['subject', 'spatial', 'temporal', 'mis'], column))
    subject_id = str(df[column[0]][sub_index])
    if task_selected == None:
        task_selected = 'spatial'
        #task_selected = 'temporal'
    task = df[task_desc_dict[task_selected]][sub_index]
    #task = df[column[1]][sub_index]
    bids_basename = make_bids_basename(
        subject=subject_id, session=None, task=task, run=None)
    bids_basename_em = make_bids_basename(
        subject=subject_id, session=None, task='EM', run=None)
    raw_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_meg.con'
    mrk_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_markers.mrk'
    elp_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_elp.elp'
    hsp_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_hsp.hsp'
    raw_em_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename_em+'_meg.con'
    print(bids_basename)
    raw = mne.io.read_raw_kit(input_fname=raw_f, mrk=mrk_f, elp=elp_f,
        hsp=hsp_f, allow_unknown_format=True)
    raw_em = mne.io.read_raw_kit(input_fname=raw_em_f, mrk=mrk_f, elp=elp_f,
        hsp=hsp_f, allow_unknown_format=True)
    events = mne.find_events(raw, 'STI 014', consecutive=True, min_duration=1,
                             initial_event=True)
    #events = mne.find_events(raw, 'STI 014', consecutive=True)
    event_id = list(np.unique(events[:,2]))
    #epochs = mne.Epochs(raw, events, event_id)
    __import__('IPython').embed()
    __import__('sys').exit()
    return raw, events, event_id, raw_em
    #return raw


def _read_raw_bids(sub_index=None, task_selected=None):
    df = read_id_task_dict()
    root_dir = "/home/foucault/data/rawdata/working_memory"
    column = ['sub_id', '0', '1', '2']
    if sub_index == None:
        sub_index = 0
    task_desc_dict = dict(zip(['subject', 'spatial', 'temporal', 'mis'], column))
    subject_id = str(df[column[0]][sub_index])
    if task_selected == None:
        task_selected = 'spatial'
        #task_selected = 'temporal'
    task = df[task_desc_dict[task_selected]][sub_index]
    #task = df[column[1]][sub_index]
    bids_basename = make_bids_basename(
        subject=subject_id, session=None, task=task, run=None)
    bids_basename_em = make_bids_basename(
        subject=subject_id, session=None, task='EM', run=None)
    raw_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_meg.con'
    mrk_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_markers.mrk'
    elp_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_elp.elp'
    hsp_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_hsp.hsp'
    raw_em_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename_em+'_meg.con'
    print(bids_basename)
    raw = mne.io.read_raw_kit(input_fname=raw_f, mrk=mrk_f, elp=elp_f,
        hsp=hsp_f, allow_unknown_format=True)
    raw_em = mne.io.read_raw_kit(input_fname=raw_em_f, mrk=mrk_f, elp=elp_f,
        hsp=hsp_f, allow_unknown_format=True)
    events = mne.find_events(raw, 'STI 014', consecutive=True, min_duration=1,
                             initial_event=True)
    #events = mne.find_events(raw, 'STI 014', consecutive=True)
    event_id = list(np.unique(events[:,2]))
    #epochs = mne.Epochs(raw, events, event_id)
    return raw, events, event_id, raw_em
    #return raw


def devp_plot_layout():
    import matplotlib.pyplot as plt
    #__import__('matplotlib').use('TkAgg')
    base_d = '/home/foucault/projects/aging_multimodality'
    biosemi_layout = mne.channels.read_layout('KIT-TW-157.lout', path=base_d)
    channels_partition = list_of_part()
    #fig, axes = plt.subplots(1, 1)
    __import__('IPython').embed()
    __import__('sys').exit()
    fig = biosemi_layout.plot(show=False, names=channels_partition[0])
    fig.set_size_inches(30,30)
    #fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
    fig.savefig('viz_sensor_layout_la.png')
    __import__('IPython').embed()
    __import__('sys').exit()
    raw, events, event_id, raw_em = devp_read_raw_bids(sub_index, task_selected)
    layout = mne.channels.find_layout(raw.info)
    __import__('IPython').embed()
    __import__('sys').exit()


def devp_epochs_filter_by_trial():
    from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)
    from mne.preprocessing import (compute_proj_ecg, compute_proj_eog)
    sub_index = 0
    task_selected='spatial'
    #task_selected='temporal'
    raw, events, event_id, raw_em = devp_read_raw_bids(sub_index, task_selected)
    #raw_em.plot_psd()
    empty_room_projs = mne.compute_proj_raw(raw_em, n_mag=3)
    # mne.viz.plot_projs_topomap(empty_room_projs, colorbar=True,
                                # info=raw_em.info)
    #projs, events = compute_proj_eog(raw, n_mag=1, no_proj=True)
    # No good epochs found for eog and ecg ssp
    #projs, events = compute_proj_ecg(raw, n_mag=1, no_proj=True)
    raw.add_proj(empty_room_projs)
    #projs, events = compute_proj_ecg(raw, n_mag=1)
    #mne.viz.plot_projs_topomap(projs, colorbar=True, info=raw.info)
    #raw.plot(proj=True)
    #raw.crop(60, 360).load_data()
    #raw.plot()
    #mag_channels = mne.pick_types(raw.info, meg='mag')
    # nothing in EEG channel
    #regexp = r'(MEG)' #STI, TRIGGER, MISC
    #regexp = r'(MEG [12][45][123]1|EEG 00.)'
    # artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
    # raw.plot(order=artifact_picks)

    # return None
    # df = read_eprime_csv(sub_index)
    # logger.info(f"indexed by event_id: 254")
    # event_index = np.where(events[:][:,2]==254)[0] #n=63
    # acc = df[df['ExperimentName'].str.contains('Spat')]['Probe.ACC']
    # acc_anno_index =  np.where(acc==0)[0]
    # marked = event_index[acc_anno_index]
    # marked_end = event_index[acc_anno_index+1]
    # anno_onset_array = events[marked][:,0] / raw.info['sfreq']
    # anno_end_array = events[marked_end][:,0] / raw.info['sfreq']
    # duration_array = anno_end_array - anno_onset_array
    # anno_onset_array -= 0.25
    # anno_end_array -= 0.02
    # description_list = ['bad']*len(anno_onset_array)
    # my_annot = mne.Annotations(onset=anno_onset_array,
                               # duration=duration_array,
                               # description=description_list)
    # raw.set_annotations(my_annot)
    # print(raw.annotations)

    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=1., h_freq=None)
    ica = ICA(n_components=30, random_state=97)
    ica.fit(filt_raw)
    #ica.exclude = [0]
    #reconst_raw = raw.copy()
    #ica.apply(reconst_raw.load_data())
    #template_eog_component = ica.get_components()[:,0]
    template_file = 'template_eog_component.csv'
    template_eog_component = np.loadtxt(template_file, delimiter=',')
    # corrmap([ica], template=template_eog_component, threshold=0.9, ch_type='mag')
    corrmap([ica], template=template_eog_component, threshold=0.9, ch_type='mag',
            label='blink', plot=False)
    ica.exclude = ica.labels_['blink']
    reconst_raw = raw.copy()
    reconst_raw.load_data().filter(l_freq=1.5, h_freq=30)
    ica.apply(reconst_raw)
    #logger.info(f"Save template_eog_component file: {template_file}")
    #np.savetxt(template_file, template_eog_component, delimiter=',')
    #raw.plot()
    #reconst_raw.plot()
    #ica.apply(raw)
    #eog_indices, eog_scores = ica.find_bads_eog(raw)
    # ica.plot_sources(raw)
    # __import__('matplotlib').use('TkAgg')
    # ica.plot_components()

    # ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    # ecg_epochs.plot_image(picks='meg', combine='mean')
    # ecg component not clear

    # eog_epochs = mne.preprocessing.create_eog_epochs(raw,  ch_name='MEG 068')
    # eog_epochs.plot_image(picks='meg', combine='mean')
    #eog_epochs.average().plot_joint()
    # epochs = mne.Epochs(raw, events, reject_by_annotation=True, preload=True)
    # epochs.plot_topo_image()
    #epochs.plot_drop_log()
    #return raw
    return reconst_raw


def dict_of_part():
    ch_quad = dict()
    ch_quad['la'] = ['MEG 001', 'MEG 002', 'MEG 003', 'MEG 004', 'MEG 005', 'MEG 006', 'MEG 007', 'MEG 008', 'MEG 009', 'MEG 010', 'MEG 011', 'MEG 012', 'MEG 013', 'MEG 014', 'MEG 015', 'MEG 016', 'MEG 017', 'MEG 018', 'MEG 022', 'MEG 023', 'MEG 025', 'MEG 026', 'MEG 028', 'MEG 029', 'MEG 030', 'MEG 031', 'MEG 032', 'MEG 033', 'MEG 034', 'MEG 035', 'MEG 036', 'MEG 037', 'MEG 038', 'MEG 039', 'MEG 040', 'MEG 041', 'MEG 042', 'MEG 043', 'MEG 044', 'MEG 045', 'MEG 046', 'MEG 047', 'MEG 048', 'MEG 049', 'MEG 056', 'MEG 058', 'MEG 059', 'MEG 061', 'MEG 063', 'MEG 068', 'MEG 094']
    ch_quad['lp'] = ['MEG 019', 'MEG 020', 'MEG 021', 'MEG 024', 'MEG 027', 'MEG 050', 'MEG 051', 'MEG 052', 'MEG 054', 'MEG 055', 'MEG 057', 'MEG 060', 'MEG 062', 'MEG 064', 'MEG 081', 'MEG 129', 'MEG 130', 'MEG 131', 'MEG 132', 'MEG 133', 'MEG 134', 'MEG 135', 'MEG 136', 'MEG 137', 'MEG 138', 'MEG 139', 'MEG 140', 'MEG 141', 'MEG 142', 'MEG 143', 'MEG 144']
    ch_quad['ra'] = ['MEG 065', 'MEG 066', 'MEG 067', 'MEG 069', 'MEG 070', 'MEG 071', 'MEG 072', 'MEG 073', 'MEG 074', 'MEG 075', 'MEG 076', 'MEG 077', 'MEG 078', 'MEG 079', 'MEG 080', 'MEG 085', 'MEG 087', 'MEG 088', 'MEG 090', 'MEG 091', 'MEG 092', 'MEG 093', 'MEG 095', 'MEG 096', 'MEG 097', 'MEG 098', 'MEG 099', 'MEG 100', 'MEG 101', 'MEG 102', 'MEG 103', 'MEG 104', 'MEG 105', 'MEG 106', 'MEG 107', 'MEG 108', 'MEG 109', 'MEG 110', 'MEG 111', 'MEG 112', 'MEG 116', 'MEG 119', 'MEG 123', 'MEG 124', 'MEG 126', 'MEG 128']
    ch_quad['rp'] = ['MEG 082', 'MEG 083', 'MEG 084', 'MEG 086', 'MEG 089', 'MEG 113', 'MEG 114', 'MEG 115', 'MEG 117', 'MEG 118', 'MEG 120', 'MEG 121', 'MEG 122', 'MEG 125', 'MEG 127', 'MEG 145', 'MEG 146', 'MEG 147', 'MEG 148', 'MEG 149', 'MEG 150', 'MEG 151', 'MEG 152', 'MEG 153', 'MEG 154', 'MEG 155', 'MEG 156', 'MEG 157']
    return ch_quad


def list_of_part():
    la = ['MEG 001', 'MEG 002', 'MEG 003', 'MEG 004', 'MEG 005', 'MEG 006', 'MEG 007', 'MEG 008', 'MEG 009', 'MEG 010', 'MEG 011', 'MEG 012', 'MEG 013', 'MEG 014', 'MEG 015', 'MEG 016', 'MEG 017', 'MEG 018', 'MEG 022', 'MEG 023', 'MEG 025', 'MEG 026', 'MEG 028', 'MEG 029', 'MEG 030', 'MEG 031', 'MEG 032', 'MEG 033', 'MEG 034', 'MEG 035', 'MEG 036', 'MEG 037', 'MEG 038', 'MEG 039', 'MEG 040', 'MEG 041', 'MEG 042', 'MEG 043', 'MEG 044', 'MEG 045', 'MEG 046', 'MEG 047', 'MEG 048', 'MEG 049', 'MEG 056', 'MEG 058', 'MEG 059', 'MEG 061', 'MEG 063', 'MEG 068', 'MEG 094']
    lp = ['MEG 019', 'MEG 020', 'MEG 021', 'MEG 024', 'MEG 027', 'MEG 050', 'MEG 051', 'MEG 052', 'MEG 054', 'MEG 055', 'MEG 057', 'MEG 060', 'MEG 062', 'MEG 064', 'MEG 081', 'MEG 129', 'MEG 130', 'MEG 131', 'MEG 132', 'MEG 133', 'MEG 134', 'MEG 135', 'MEG 136', 'MEG 137', 'MEG 138', 'MEG 139', 'MEG 140', 'MEG 141', 'MEG 142', 'MEG 143', 'MEG 144']
    ra = ['MEG 065', 'MEG 066', 'MEG 067', 'MEG 069', 'MEG 070', 'MEG 071', 'MEG 072', 'MEG 073', 'MEG 074', 'MEG 075', 'MEG 076', 'MEG 077', 'MEG 078', 'MEG 079', 'MEG 080', 'MEG 085', 'MEG 087', 'MEG 088', 'MEG 090', 'MEG 091', 'MEG 092', 'MEG 093', 'MEG 095', 'MEG 096', 'MEG 097', 'MEG 098', 'MEG 099', 'MEG 100', 'MEG 101', 'MEG 102', 'MEG 103', 'MEG 104', 'MEG 105', 'MEG 106', 'MEG 107', 'MEG 108', 'MEG 109', 'MEG 110', 'MEG 111', 'MEG 112', 'MEG 116', 'MEG 119', 'MEG 123', 'MEG 124', 'MEG 126', 'MEG 128']
    rp = ['MEG 082', 'MEG 083', 'MEG 084', 'MEG 086', 'MEG 089', 'MEG 113', 'MEG 114', 'MEG 115', 'MEG 117', 'MEG 118', 'MEG 120', 'MEG 121', 'MEG 122', 'MEG 125', 'MEG 127', 'MEG 145', 'MEG 146', 'MEG 147', 'MEG 148', 'MEG 149', 'MEG 150', 'MEG 151', 'MEG 152', 'MEG 153', 'MEG 154', 'MEG 155', 'MEG 156', 'MEG 157']
    return (la, lp, ra, rp)


def epochs_filter_by_trial(sub_index=None, task_selected=None):
    from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)
    from mne.preprocessing import (compute_proj_ecg, compute_proj_eog)
    if sub_index == None:
        sub_index = 0
    if task_selected == None:
        task_selected = 'spatial'
        #task_selected = 'temporal'
    id_list, exclude_list, data_concated_dict = define_id_list()
    if str(id_list[sub_index]) in data_concated_dict.keys():
        raw, events, event_id, raw_em =\
            data_concated_dict[str(id_list[sub_index])](task_selected)
    else:
        raw, events, event_id, raw_em = _read_raw_bids(sub_index, task_selected)
    def plot_layout():
        channels_dict = {0: 'la', 1: 'lp', 2: 'ra', 3: 'rp'} # corr to partition dict
        i_key = 3
        channels_partition = list_of_part()
        pick_ch = mne.pick_channels(raw.ch_names, channels_partition[i_key])
        layout = mne.channels.find_layout(raw.info)
        fig = layout.plot(show=False, picks=pick_ch)
        fig.set_size_inches(15,15)
        logger.info(f'saving viz_sensor_layout_{channels_dict[i_key]}.png')
        fig.savefig(f'viz_sensor_layout_{channels_dict[i_key]}.png')
    #plot_layout()
    empty_room_projs = mne.compute_proj_raw(raw_em, n_mag=3)
    raw.add_proj(empty_room_projs)

    # Read eprime data
    df = read_eprime_csv(sub_index)
    indexed_event_id = 253
    logger.info(f"indexed by event_id: {indexed_event_id}")
    event_index = np.where(events[:][:,2]==indexed_event_id)[0] #n=63
    task_desc_dict = dict(zip(['spatial', 'temporal'], ['Spat', 'Temp']))
    acc = df[df['ExperimentName'].str.contains(task_desc_dict[task_selected])]['Probe.ACC']
    acc_anno_index =  np.where(acc==0)[0]
    if acc_anno_index.size == 0:
        acc_anno_index = np.array([len(event_index)-1])
    # __import__('IPython').embed()
    # __import__('sys').exit()
    marked = event_index[acc_anno_index]
    anno_onset_array = events[marked][:,0] / raw.info['sfreq']
    if acc_anno_index[-1] == len(event_index)-1:
        acc_anno_index = np.delete(acc_anno_index, -1)
        marked_end = event_index[acc_anno_index+1]
        marked_end = np.append(marked_end, len(events)-1)
        anno_end_array = events[marked_end][:,0] / raw.info['sfreq']
    else:
        marked_end = event_index[acc_anno_index+1]
        anno_end_array = events[marked_end][:,0] / raw.info['sfreq']
    duration_array = anno_end_array - anno_onset_array
    anno_onset_array -= 0.25
    anno_end_array -= 0.02
    description_list = ['bad']*len(anno_onset_array)
    my_annot = mne.Annotations(onset=anno_onset_array,
                               duration=duration_array,
                               description=description_list)
    raw.set_annotations(my_annot)

    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=1., h_freq=None)
    ica = ICA(n_components=30, random_state=97)
    ica.fit(filt_raw)
    #ica.exclude = [0]
    #reconst_raw = raw.copy()
    #ica.apply(reconst_raw.load_data())
    #template_eog_component = ica.get_components()[:,0]
    template_file = 'template_eog_component.csv'
    template_eog_component = np.loadtxt(template_file, delimiter=',')
    # corrmap([ica], template=template_eog_component, threshold=0.9, ch_type='mag')
    corrmap([ica], template=template_eog_component, threshold=0.7, ch_type='mag',
            label='blink', plot=False)
    # __import__('IPython').embed()
    # __import__('sys').exit()
    if hasattr(ica, "lables_"):
        ica.exclude = ica.labels_['blink']
    reconst_raw = raw.copy()
    reconst_raw.load_data().filter(l_freq=1.5, h_freq=50)
    #reconst_raw.load_data().filter(l_freq=1.5, h_freq=30)
    ica.apply(reconst_raw)
    #events = mne.find_events(raw, 'STI 014', consecutive=True)
    # event_id = list(np.unique(events[:,2]))
    # epochs = mne.Epochs(reconst_raw, events, event_id, reject=dict(mag=4000e-13))
    # __import__('IPython').embed()
    # __import__('sys').exit()
    #logger.info(f"Save template_eog_component file: {template_file}")
    #np.savetxt(template_file, template_eog_component, delimiter=',')
    #raw.plot()
    #reconst_raw.plot()
    #ica.apply(raw)
    #eog_indices, eog_scores = ica.find_bads_eog(raw)
    # ica.plot_sources(raw)
    # __import__('matplotlib').use('TkAgg')
    # ica.plot_components()

    # ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    # ecg_epochs.plot_image(picks='meg', combine='mean')
    # ecg component not clear

    # eog_epochs = mne.preprocessing.create_eog_epochs(raw,  ch_name='MEG 068')
    # eog_epochs.plot_image(picks='meg', combine='mean')
    #eog_epochs.average().plot_joint()
    # epochs = mne.Epochs(raw, events, reject_by_annotation=True, preload=True)
    # epochs.plot_topo_image()
    #epochs.plot_drop_log()
    #return raw
    return reconst_raw, events


def save_list_pkl(_list, _file, logger=None):
    if logger is None:
        logger = Logger()
    root_dir = Path('/home/foucault/data/derivatives/working_memory/intermediate')
    file_w = root_dir.joinpath(_file)
    logger.info('save file')
    logger.info(f'{file_w.as_posix()}')
    with open(file_w.as_posix(), 'wb') as f:
        pickle.dump(_list, f)


def load_list_pkl(_file, logger=None):
    if logger is None:
        logger = Logger()
    root_dir = Path('/home/foucault/data/derivatives/working_memory/intermediate')
    file_r = root_dir.joinpath(_file)
    logger.info('load file')
    logger.info(f'{file_r.as_posix()}')
    with open(file_r.as_posix(), 'rb') as f:
        _list = pickle.load(f)
    return _list


def additional_loop():
    print('Staring test sub data')
    sub_index = 0
    id_list, exclude_list = define_id_list()
    # 40?
    for i in range(1):
        i = 40
    #for i in range(45,len(id_list)):
        sub_index = i
        print(f"sub_index: {i}")
        if str(id_list[sub_index]) in exclude_list:
            print(f"excluding id:{id_list[sub_index]}")
            continue
        raw_, events = epochs_filter_by_trial(sub_index, task_selected)
    print("Testing Done!")


def plot_gfp(sub_index=None, task_selected=None):
    """ old version; devp; rename to build_individual_frequency_map_pkl """
    from mne.baseline import rescale
    from mne.stats import bootstrap_confidence_interval
    __import__('matplotlib').use('TkAgg')
    import matplotlib.pyplot as plt
    iter_freqs = [
        ('Theta', 4, 7),
        ('Alpha', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)
    ]
    # set epoching parameters
    event_id, tmin, tmax = 253, -0.1, 2
    #baseline = (-0.1, 0)
    baseline = None
    if sub_index == None:
        sub_index = 0
    if task_selected == None:
        #task_selected='spatial'
        task_selected='temporal'
    # for test data
    #additional_loop()
    #return None

    raw_, events = epochs_filter_by_trial(sub_index, task_selected)
    frequency_map = list()
    for band, fmin, fmax in iter_freqs:
        # (re)load the data to save memory
        raw = raw_.copy()
        raw.pick_types(meg='mag')  # we just look at gradiometers

        # bandpass filter
        raw.filter(fmin, fmax, n_jobs=8,  # use more jobs to speed up.
                   l_trans_bandwidth=1,  # make sure filter params are the same
                   h_trans_bandwidth=1)  # in each band and skip "auto" option.

        # epoch
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline,
                            reject=dict(mag=1.5e-12),
                            preload=True)
        # remove evoked response
        epochs.subtract_evoked()
        # get analytic signal (envelope)
        # __import__('IPython').embed()
        # __import__('sys').exit()
        if epochs.events.size != 0:
            epochs.apply_hilbert(envelope=True)
        frequency_map.append(((band, fmin, fmax), epochs.average()))
        del epochs
    id_list, exclude_list, data_concated_dict = define_id_list()
    _file = f"sub-{id_list[sub_index]}_{task_selected}_frequency_map.pkl"
    save_list_pkl(frequency_map, _file)
    #frequency_map = load_list_pkl(_file)
    del raw
    # Helper function for plotting spread
    def stat_fun(x):
        """Return sum of squares."""
        return np.sum(x ** 2, axis=0)
    def _plot_gfp():
        # Plot
        fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
        colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
        for ((freq_name, fmin, fmax), average), color, ax in zip(
                frequency_map, colors, axes.ravel()[::-1]):
            times = average.times * 1e3
            gfp = np.sum(average.data ** 2, axis=0)
            gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
            ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
            ax.axhline(0, linestyle='--', color='grey', linewidth=2)
            ci_low, ci_up = bootstrap_confidence_interval(average.data, random_state=0,
                                                        stat_fun=stat_fun)
            ci_low = rescale(ci_low, average.times, baseline=(None, 0))
            ci_up = rescale(ci_up, average.times, baseline=(None, 0))
            ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=color, alpha=0.3)
            ax.grid(True)
            ax.set_ylabel('GFP')
            ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                        xy=(0.95, 0.8),
                        horizontalalignment='right',
                        xycoords='axes fraction')
            ax.set_xlim(-100, 2000)
            #ax.set_xlim(-1000, 3000)
        axes.ravel()[-1].set_xlabel('Time [ms]')
        #w_png = f"gfp_{task_selected}_b_s1.png"
        w_png = f"gfp_{task_selected}_s1.png"
        logger.info(f"saving file: {w_png}")
        fig.savefig(w_png)
        #_plot_gfp()


def build_individual_frequency_map_pkl(sub_index=None, task_selected=None):
    """ build_individual_frequency_map_pkl """
    from mne.baseline import rescale
    from mne.stats import bootstrap_confidence_interval
    __import__('matplotlib').use('TkAgg')
    import matplotlib.pyplot as plt
    iter_freqs = [
        ('Theta', 4, 7),
        ('Alpha', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)
    ]
    # set epoching parameters
    event_id, tmin, tmax = 253, -0.1, 2
    #baseline = (-0.1, 0)
    baseline = None
    if sub_index == None:
        sub_index = 0
    if task_selected == None:
        #task_selected='spatial'
        task_selected='temporal'

    raw_, events = epochs_filter_by_trial(sub_index, task_selected)
    frequency_map = list()
    for band, fmin, fmax in iter_freqs:
        # (re)load the data to save memory
        raw = raw_.copy()
        raw.pick_types(meg='mag')  # we just look at gradiometers

        # bandpass filter
        raw.filter(fmin, fmax, n_jobs=8,  # use more jobs to speed up.
                   l_trans_bandwidth=1,  # make sure filter params are the same
                   h_trans_bandwidth=1)  # in each band and skip "auto" option.

        # epoch
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline,
                            reject=dict(mag=1.5e-12),
                            preload=True)
        # remove evoked response
        epochs.subtract_evoked()
        # get analytic signal (envelope)
        if epochs.events.size != 0:
            epochs.apply_hilbert(envelope=True)
        frequency_map.append(((band, fmin, fmax), epochs.average()))
        del epochs
    id_list, exclude_list, data_concated_dict = define_id_list()
    _file = f"sub-{id_list[sub_index]}_{task_selected}_frequency_map.pkl"
    save_list_pkl(frequency_map, _file)
    # __import__('IPython').embed()
    # __import__('sys').exit()


def devp_build_grand_frequency_maps_pkl(sub_index=None, task_selected=None):
    import copy
    task_selected = ['spatial', 'temporal']
    id_list, exclude_list, data_concated_dict = define_id_list()
    frequency_maps = list()
    for task in task_selected:
        grand_frequency_map = list()
        # if task == 'temporal':
            # continue
        count = 0
        # for i in range(1):
            # sub_index = 16
        # for sub_index in range(0, 2):
            #sub_index = 16
        for sub_index in range(0, len(id_list)):
            logger.info(f"sub_index: {sub_index}; sub-{id_list[sub_index]}")
            logger.info(f"task: {task}")
            if str(id_list[sub_index]) in exclude_list:
                logger.info(f"excluding id:{id_list[sub_index]}")
                continue
            _file = f"sub-{id_list[sub_index]}_{task}_frequency_map.pkl"
            frequency_map = load_list_pkl(_file)
            # print("add this")
            # print(frequency_map[0][1].data[0][0])
            frequency_maps.append(frequency_map)
            count += 1
            if count == 1:
                grand_frequency_map = copy.deepcopy(frequency_map)
                for i, f_band in enumerate(frequency_map):
                    # reset nave to n=1 for grand average
                    grand_frequency_map[i][1].nave = 1
                continue
            for i, f_band in enumerate(frequency_map):
                if np.isnan(f_band[1].data[0][0]):
                    continue
                grand_frequency_map[i][1].data += f_band[1].data
                grand_frequency_map[i][1].nave += 1
                # print(grand_frequency_map)
                # print("sum")
                # print(grand_frequency_map[i][1].data[0][0])
        # get average
        for i in range(len(grand_frequency_map)):
            grand_frequency_map[i][1].data /= grand_frequency_map[i][1].nave
        _file = f"grand_{task}_frequency_map.pkl"
        save_list_pkl(grand_frequency_map, _file)
        logger.info(f"#participants in task_{task}: {count}")


def append_grand_frequency_map(id_list, task):
    import copy
    _id_list, exclude_list, data_concated_dict = define_id_list()
    grand_frequency_map = list()
    #frequency_maps = list()
    # if task == 'temporal':
        # continue
    count = 0
    # for i in range(1):
        # sub_index = 16
    # for sub_index in range(0, 2):
        #sub_index = 16
    for sub_index in range(0, len(id_list)):
        logger.info(f"sub_index: {sub_index}; sub-{id_list[sub_index]}")
        logger.info(f"task: {task}")
        if str(id_list[sub_index]) in exclude_list:
            logger.info(f"excluding id:{id_list[sub_index]}")
            continue
        _file = f"sub-{id_list[sub_index]}_{task}_frequency_map.pkl"
        frequency_map = load_list_pkl(_file)
        #frequency_maps.append(frequency_map)
        count += 1
        if count == 1:
            grand_frequency_map = copy.deepcopy(frequency_map)
            for i, f_band in enumerate(frequency_map):
                if np.isnan(frequency_map[i][1].data).any():
                    grand_frequency_map[i][1].data = \
                        np.zeros(frequency_map[i][1].data.shape)
                    grand_frequency_map[i][1].nave = 0
                else:
                    # reset nave to n=1 for grand average
                    grand_frequency_map[i][1].nave = 1
            continue
        for i, f_band in enumerate(frequency_map):
            if np.isnan(f_band[1].data[0][0]):
                continue
            grand_frequency_map[i][1].data += f_band[1].data
            grand_frequency_map[i][1].nave += 1
    # get average
    for i in range(len(grand_frequency_map)):
        grand_frequency_map[i][1].data /= grand_frequency_map[i][1].nave
    logger.info(f"#participants in task_{task}: {count}")
    return grand_frequency_map


def calculate_gfp(evoked):
    times = evoked.times * 1e3
    gfp = np.sum(evoked.data ** 2, axis=0)
    gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
    return gfp


def save_individual_frequency_gfp_aging_hemisphere_pkl(id_list, task):
    """As function name """
    _id_list, exclude_list, data_concated_dict = define_id_list()
    chs_name = ['lr', 'ap']
    #frequency_maps = list()
    # if task == 'temporal':
        # continue
    # for i in range(1):
        # sub_index = 16
    # for sub_index in range(0, 2):
        #sub_index = 16
    for sub_index in range(0, len(id_list)):
        logger.info(f"sub_index: {sub_index}; sub-{id_list[sub_index]}")
        logger.info(f"task: {task}")
        if str(id_list[sub_index]) in exclude_list:
            logger.info(f"excluding id:{id_list[sub_index]}")
            continue
        file_r = f"sub-{id_list[sub_index]}_{task}_frequency_map.pkl"
        chs_lr_pair, chs_ap_pair = ch_di_list()
        for i_chs_pair, chs_pairs in enumerate([chs_lr_pair, chs_ap_pair]):
            frequency_gfp = list()
            for i_chs, chs in enumerate(chs_pairs):
                frequency_map = load_list_pkl(file_r)
                for i in range(len(frequency_map)):
                    frequency_map[i][1].pick_channels(chs)
                    """ frequency_gfp: ['l'*4 'r'*4] """
                    """ frequency_gfp: ['a'*4 'p'*4] """
                    frequency_gfp.append(calculate_gfp(frequency_map[i][1]))
                    #print(f"{frequency_map[i][1]}")
            # __import__('IPython').embed()
            # __import__('sys').exit()
            _file = f"sub-{id_list[sub_index]}_{task}_{chs_name[i_chs_pair]}_frequency_gfp.pkl"
            save_list_pkl(frequency_gfp, _file)
            # __import__('IPython').embed()
            # __import__('sys').exit()


def append_individual_frequency_gfp_aging_hemisphere_pkl(id_list, task):
    _id_list, exclude_list, data_concated_dict = define_id_list()
    iter_freqs = [
        ('Theta', 4, 7),
        ('Alpha', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)]
    gfp_dict = {i: [[], []] for i in iter_freqs}
    chs_name = ['lr', 'ap']
    chs_lr_pair, chs_ap_pair = ch_di_list()
    ch_1 = {ch: [] for ch in ['l', 'r']}
    ch_2 = {ch: [] for ch in ['a', 'p']}
    _chs = {chs: copy.deepcopy((ch_1, ch_2)[i]) for i, chs in enumerate(chs_name)}
    _band = {band: copy.deepcopy(_chs) for band in iter_freqs}
    chs_dict = {chs: [] for chs in chs_name}
    _gfp_pair_dict = {band: copy.deepcopy(chs_dict) for band in iter_freqs}
    gfp_pair_dict = {band: copy.deepcopy(chs_dict) for band in iter_freqs}
    #gfp_pair_dict = {i: [[], []] for i in iter_freqs}
    #_task = {task: _chs for task in task_selected}
    for i_chs_pair, chs_pairs in enumerate(zip(chs_lr_pair, chs_ap_pair)):
        count_sub = 0
        # if task == 'temporal':
            # continue
        # for i in range(1):
            # sub_index = 16
        # for sub_index in range(0, 2):
            #sub_index = 16
        _gfp_dict = {i: [[], []] for i in iter_freqs}
        print("_gfp_pair_dict")
        print(f"{_gfp_pair_dict}")
        for sub_index in range(0, len(id_list)):
            logger.info(f"sub_index: {sub_index}; sub-{id_list[sub_index]}")
            logger.info(f"task: {task}")
            if str(id_list[sub_index]) in exclude_list:
                logger.info(f"excluding id:{id_list[sub_index]}")
                continue
            file_r = (f"sub-{id_list[sub_index]}_{task}_{chs_name[i_chs_pair]}"
                      "_frequency_gfp.pkl")
            logger.info("loading:")
            logger.info(f"{file_r}")
            frequency_gfp = load_list_pkl(file_r)
            count_sub += 1
            for i_band, band in enumerate(gfp_dict.keys()):
                gfp_dict[band][0].append(frequency_gfp[i_band])
                gfp_dict[band][1].append(frequency_gfp[i_band+4])
                _gfp_dict[band][0].append(frequency_gfp[i_band])
                _gfp_dict[band][1].append(frequency_gfp[i_band+4])
                _gfp_pair_dict[band][chs_name[i_chs_pair]].append(
                    frequency_gfp[i_band] - frequency_gfp[i_band+4])
        for i_band, band in enumerate(gfp_dict.keys()):
            gfp_pair_dict[band][chs_name[i_chs_pair]] = np.array(_gfp_pair_dict[band]
                                                       [chs_name[i_chs_pair]])
            for i_key, key in enumerate(_band[band][chs_name[i_chs_pair]].keys()):
                _band[band][chs_name[i_chs_pair]][key] = np.array(_gfp_dict[band][i_key])
    # print("gfp_dict here")
    # __import__('IPython').embed()
    # __import__('sys').exit()
    logger.info(f"number of particiapnts: {count_sub}")
    return _band, gfp_pair_dict, count_sub


def build_grand_frequency_maps_pkl(sub_index=None, task_selected=None):
    task_selected = ['spatial', 'temporal']
    id_list, exclude_list, data_concated_dict = define_id_list()
    aging_dict = define_id_by_aging_dict()
    for task in task_selected:
        grand_frequency_map = append_grand_frequency_map(id_list, task)
        _file = f"grand_{task}_frequency_map.pkl"
        save_list_pkl(grand_frequency_map, _file)


def build_aging_grand_frequency_maps_pkl(sub_index=None, task_selected=None):
    """Maps divided by young and elder groups """
    task_selected = ['spatial', 'temporal']
    aging_variable = ['young', 'elder']
    id_list, exclude_list, data_concated_dict = define_id_list()
    aging_dict = define_id_by_aging_dict()
    for i in aging_variable:
        if i == 'old':
            continue
        for task in task_selected:
            # if task == 'spatial':
                # continue
            grand_frequency_map = append_grand_frequency_map(aging_dict[i], task)
            _file = f"{i}_grand_{task}_frequency_map.pkl"
            save_list_pkl(grand_frequency_map, _file)


def save_individual_aging_frequency_gfp_dict(sub_index=None, task_selected=None):
    """Maps divided by young and elder groups """
    """Note: elder part has nan values """
    task_selected = ['spatial', 'temporal']
    aging_variable = ['young', 'elder']
    chs_name = ['lr', 'ap']
    id_list, exclude_list, data_concated_dict = define_id_list()
    aging_dict = define_id_by_aging_dict()
    gfp_pair_task_dict = {i: [] for i in task_selected}
    gfp_pair_aging_dict = {i: [] for i in aging_variable}
    _task = {task: {} for task in task_selected}
    gfp_aging_task_chs_dict = {aging: copy.deepcopy(_task) for aging in aging_variable}
    for aging in aging_variable:
        if aging == 'elder':
            continue
        logger.info(f"loop: {aging}")
        for task in task_selected:
            logger.info(f"loop: {task}")
            # if task == 'spatial':
                # continue
            save_individual_frequency_gfp_aging_hemisphere_pkl(aging_dict[aging], task)


def build_individual_aging_frequency_gfp_dict(sub_index=None, task_selected=None):
    """Maps divided by young and elder groups """
    """Note: elder part has nan values """
    task_selected = ['spatial', 'temporal']
    aging_variable = ['young', 'elder']
    chs_name = ['lr', 'ap']
    id_list, exclude_list, data_concated_dict = define_id_list()
    aging_dict = define_id_by_aging_dict()
    gfp_pair_task_dict = {i: [] for i in task_selected}
    gfp_pair_aging_dict = {i: [] for i in aging_variable}
    _task = {task: {} for task in task_selected}
    gfp_aging_task_chs_dict = {aging: copy.deepcopy(_task) for aging in aging_variable}
    for aging in aging_variable:
        if aging == 'elder':
            continue
        logger.info(f"loop: {aging}")
        for task in task_selected:
            logger.info(f"loop: {task}")
            # if task == 'spatial':
                # continue
            # save_individual_frequency_gfp_aging_hemisphere_pkl(aging_dict[aging], task)
            # print("save_individual_frequency_gfp_aging_hemisphere_pkl")
            # __import__('IPython').embed()
            # __import__('sys').exit()
            _band, gfp_pair_dict, count_sub = \
                append_individual_frequency_gfp_aging_hemisphere_pkl(aging_dict[aging], task)
            gfp_pair_task_dict[task].append(gfp_pair_dict)
            # gfp_pair_task_dict[task][0][('Theta', 4, 7)][:count_sub]
            # only young group
            gfp_aging_task_chs_dict[aging][task] = _band
        gfp_pair_aging_dict[aging] = gfp_pair_task_dict
    #epochs_power = gfp_pair_task_dict[task][0][('Theta', 4, 7)][:count_sub]
    #epochs_power = gfp_pair_task_dict[task][0][('Theta', 4, 7)]
    # print("here")
    # __import__('IPython').embed()
    # __import__('sys').exit()
    return gfp_pair_aging_dict, gfp_aging_task_chs_dict


def calculate_permutation_cluster_1samp_test(epochs_power):
    from mne.stats import permutation_cluster_1samp_test
    #threshold = 2.5
    #threshold = 3.0
    threshold = 2.8
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_1samp_test(epochs_power,
            n_permutations=1000, threshold=threshold, tail=0)
    return T_obs, clusters, cluster_p_values, H0


def calculate_permutation_cluster_test(epochs_power):
    from mne.stats import permutation_cluster_test
    #threshold = 2.5
    threshold = 6.0
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test(epochs_power,
            n_permutations=1000, threshold=threshold, tail=0)
    return T_obs, clusters, cluster_p_values, H0


def collect_permutation_cluster_1samp_test():
    """Organizing aging: task: channels """
    iter_freqs = [
        ('Theta', 4, 7),
        ('Alpha', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)]
    task_selected = ['spatial', 'temporal']
    aging_variable = ['young', 'elder']
    chs_name = ['lr', 'ap']
    gfp_dict = {band: [] for band in iter_freqs}
    chs_dict = {chs: copy.deepcopy(gfp_dict) for chs in chs_name}
    task_dict = {task: copy.deepcopy(chs_dict) for task in task_selected}
    collection_dict = {aging: copy.deepcopy(task_dict) for aging in aging_variable}
    gfp_pair_aging_dict, gfp_aging_task_chs_dict =\
        build_individual_aging_frequency_gfp_dict()
    for aging in collection_dict.keys():
        if aging == 'elder':
            continue
        logger.info(f"loop: {aging}")
        for task in task_dict.keys():
            logger.info(f"loop: {task}")
            for chs in chs_dict.keys():
                logger.info(f"loop: {chs}")
                for band in gfp_dict.keys():
                    logger.info(f"loop: {band}")
                    epochs_power = \
                        gfp_pair_aging_dict[aging][task][0][band][chs]
                    #T_obs, clusters, cluster_p_values, H0 = \
                    collection_dict[aging][task][chs][band] = \
                        calculate_permutation_cluster_1samp_test(epochs_power)
    # print("1samp_test")
    # __import__('IPython').embed()
    # __import__('sys').exit()
    return collection_dict


def collect_permutation_cluster_test():
    """Organizing aging: task: channels """
    iter_freqs = [
        ('Theta', 4, 7),
        ('Alpha', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)]
    task_selected = ['spatial', 'temporal']
    aging_variable = ['young', 'elder']
    chs_name = ['lr', 'ap']
    gfp_dict = {band: [] for band in iter_freqs}
    chs_dict = {chs: copy.deepcopy(gfp_dict) for chs in chs_name}
    task_dict = {task: copy.deepcopy(chs_dict) for task in task_selected}
    collection_dict = {aging: copy.deepcopy(task_dict) for aging in aging_variable}
    gfp_pair_aging_dict, gfp_aging_task_chs_dict =\
        build_individual_aging_frequency_gfp_dict()
    for aging in collection_dict.keys():
        if aging == 'elder':
            continue
        logger.info(f"loop: {aging}")
        for task in task_dict.keys():
            logger.info(f"loop: {task}")
            for chs in chs_dict.keys():
                logger.info(f"loop: {chs}")
                for band in gfp_dict.keys():
                    logger.info(f"loop: {band}")
                    cons = []
                    for key in gfp_aging_task_chs_dict[aging][task][band][chs].keys():
                        logger.info(f"loop: {key}")
                        cons.append(gfp_aging_task_chs_dict[aging][task][band][chs][key])
                    #T_obs, clusters, cluster_p_values, H0 = \
                    collection_dict[aging][task][chs][band] = \
                        calculate_permutation_cluster_test(cons)
    # print("permut_test")
    # __import__('IPython').embed()
    # __import__('sys').exit()
    return collection_dict


def collect_ttest_1samp():
    """Organizing aging: task: channels """
    from scipy import stats
    iter_freqs = [
        ('Theta', 4, 7),
        ('Alpha', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)]
    task_selected = ['spatial', 'temporal']
    aging_variable = ['young', 'elder']
    chs_name = ['lr', 'ap']
    gfp_dict = {band: copy.deepcopy([]) for band in iter_freqs}
    chs_dict = {chs: copy.deepcopy(gfp_dict) for chs in chs_name}
    task_dict = {task: copy.deepcopy(chs_dict) for task in task_selected}
    collection_dict = {aging: copy.deepcopy(task_dict) for aging in aging_variable}
    gfp_pair_aging_dict, gfp_aging_task_chs_dict =\
        build_individual_aging_frequency_gfp_dict()
    for aging in collection_dict.keys():
        if aging == 'elder':
            continue
        logger.info(f"loop: {aging}")
        for task in task_dict.keys():
            logger.info(f"loop: {task}")
            for chs in chs_dict.keys():
                logger.info(f"loop: {chs}")
                for band in gfp_dict.keys():
                    logger.info(f"loop: {band}")
                    epochs_power = \
                        gfp_pair_aging_dict[aging][task][0][band][chs]
                    # Ttest_1sampResult(statistic, pvalue)
                    collection_dict[aging][task][chs][band] = \
                        stats.ttest_1samp(epochs_power, 0, axis=0)
    # pvalue = collection_dict[aging]['temporal']['lr'][('Theta', 4, 7)].pvalue
    # pvalue[pvalue < .05]
    # __import__('IPython').embed()
    # __import__('sys').exit()
    return collection_dict


def plot_time_frequency():
    pass


def devp_plot_grand_gfp():
    # Plot
    from mne.baseline import rescale
    from mne.stats import bootstrap_confidence_interval
    __import__('matplotlib').use('TkAgg')
    import matplotlib.pyplot as plt
    def stat_fun(x):
        """Return sum of squares."""
        return np.sum(x ** 2, axis=0)
    task_selected = ['spatial', 'temporal']
    ch_quad_dict = dict_of_part()
    for task in task_selected:
        _file = f"grand_{task}_frequency_map.pkl"
        frequency_map = load_list_pkl(_file)
        fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
        colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
        for ((freq_name, fmin, fmax), average), color, ax in zip(
                frequency_map, colors, axes.ravel()[::-1]):
            times = average.times * 1e3
            gfp = np.sum(average.data ** 2, axis=0)
            gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
            ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
            ax.axhline(0, linestyle='--', color='grey', linewidth=2)
            ci_low, ci_up = bootstrap_confidence_interval(average.data, random_state=0,
                                                        stat_fun=stat_fun)
            ci_low = rescale(ci_low, average.times, baseline=(None, 0))
            ci_up = rescale(ci_up, average.times, baseline=(None, 0))
            ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=color, alpha=0.3)
            ax.grid(True)
            ax.set_ylabel('GFP')
            ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                        xy=(0.95, 0.8),
                        horizontalalignment='right',
                        xycoords='axes fraction')
            ax.set_xlim(-100, 2000)
            #ax.set_xlim(-1000, 3000)
        axes.ravel()[-1].set_xlabel('Time [ms]')
        #w_png = f"gfp_{task_selected}_b_s1.png"
        w_png = f"grand_gfp_{task}.png"
        logger.info(f"saving file: {w_png}")
        fig.savefig(w_png)
        #_plot_gfp()


def ch_di_list():
    import copy
    ch_quad_dict = dict_of_part()
    ch_l_list = copy.deepcopy(ch_quad_dict['la'])
    ch_l_list.extend(ch_quad_dict['lp'])
    ch_r_list = copy.deepcopy(ch_quad_dict['ra'])
    ch_r_list.extend(ch_quad_dict['rp'])
    ch_a_list = copy.deepcopy(ch_quad_dict['la'])
    ch_a_list.extend(ch_quad_dict['ra'])
    ch_p_list = copy.deepcopy(ch_quad_dict['lp'])
    ch_p_list.extend(ch_quad_dict['rp'])
    return (ch_l_list, ch_r_list), (ch_a_list, ch_p_list)


def plot_grand_gfp(task, file_r, aging_label=None):
    from mne.baseline import rescale
    from mne.stats import bootstrap_confidence_interval
    __import__('matplotlib').use('TkAgg')
    import matplotlib.pyplot as plt
    def stat_fun(x):
        """Return sum of squares."""
        return np.sum(x ** 2, axis=0)
    chs_lr_pair, chs_ap_pair = ch_di_list()
    figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
    for i_chs_pair, chs_pairs in enumerate(zip(chs_lr_pair, chs_ap_pair)):
        chs_name = ['lr', 'ap']
        fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
        colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
        colors = colors[[0,2],:] # take 0:green, 2:blue
        for i_chs, chs in enumerate(chs_pairs):
            frequency_map = load_list_pkl(file_r)
            for i in range(len(frequency_map)):
                frequency_map[i][1].pick_channels(chs)
                print(f"{frequency_map[i][1]}")
            for ((freq_name, fmin, fmax), average), ax in zip(
                    frequency_map, axes.ravel()[::-1]):
                times = average.times * 1e3
                gfp = np.sum(average.data ** 2, axis=0)
                gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
                ax.plot(times, gfp, label=freq_name, color=colors[i_chs], linewidth=2.5)
                ax.axhline(0, linestyle='--', color='grey', linewidth=2)
                ci_low, ci_up = bootstrap_confidence_interval(average.data, random_state=0,
                                                            stat_fun=stat_fun)
                ci_low = rescale(ci_low, average.times, baseline=(None, 0))
                ci_up = rescale(ci_up, average.times, baseline=(None, 0))
                ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=colors[i_chs], alpha=0.3)
                ax.grid(True)
                ax.set_ylabel('GFP')
                ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                            xy=(0.95, 0.8),
                            horizontalalignment='right',
                            xycoords='axes fraction')
                ax.set_xlim(-100, 2000)
            axes.ravel()[-1].set_xlabel('Time [ms]')
        if aging_label:
            w_png = figures_d.joinpath(f"{aging_label}_grand_gfp_{task}_{chs_name[i_chs_pair]}_dpi.png")
        else:
            w_png = figures_d.joinpath(f"grand_gfp_{task}_{chs_name[i_chs_pair]}_dpi.png")
        logger.info("saving file:")
        logger.info(f"{w_png.as_posix()}")
        fig.savefig(w_png, dpi=300)


def plot_grand_gfp_contrast_spatial_temporal(file_rs, aging_label=None):
    from mne.baseline import rescale
    from mne.stats import bootstrap_confidence_interval
    __import__('matplotlib').use('TkAgg')
    import matplotlib.pyplot as plt
    task_selected = ['spatial', 'temporal']
    def stat_fun(x):
        """Return sum of squares."""
        return np.sum(x ** 2, axis=0)
    figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
    for i_file_r, file_r in enumerate(file_rs):
        fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
        colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
        colors = colors[[0,2],:] # take 0:green, 2:blue
        frequency_map = load_list_pkl(file_r)
        for ((freq_name, fmin, fmax), average), ax in zip(
                frequency_map, axes.ravel()[::-1]):
            times = average.times * 1e3
            gfp = np.sum(average.data ** 2, axis=0)
            gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
            ax.plot(times, gfp, label=freq_name, color=colors[i_file_r], linewidth=2.5)
            ax.axhline(0, linestyle='--', color='grey', linewidth=2)
            ci_low, ci_up = bootstrap_confidence_interval(average.data, random_state=0,
                                                          stat_fun=stat_fun)
            ci_low = rescale(ci_low, average.times, baseline=(None, 0))
            ci_up = rescale(ci_up, average.times, baseline=(None, 0))
            ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=colors[i_file_r], alpha=0.3)
            ax.grid(True)
            ax.set_ylabel('GFP')
            ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                        xy=(0.95, 0.8),
                        horizontalalignment='right',
                        xycoords='axes fraction')
            ax.set_xlim(-100, 2000)
        axes.ravel()[-1].set_xlabel('Time [ms]')
    if aging_label:
        w_png = figures_d.joinpath(f"{aging_label}_grand_gfp_contrast_spatial_temporal_dpi.png")
    else:
        w_png = figures_d.joinpath(f"grand_gfp_contrast_spatial_temporal_dpi.png")
    logger.info("saving file:")
    logger.info(f"{w_png.as_posix()}")
    fig.savefig(w_png, dpi=300)


def plot_grand_sig_clusterp_gfp(task, file_r, aging_label=None):
    from mne.baseline import rescale
    from mne.stats import bootstrap_confidence_interval
    __import__('matplotlib').use('TkAgg')
    import matplotlib.pyplot as plt
    def stat_fun(x):
        """Return sum of squares."""
        return np.sum(x ** 2, axis=0)
    chs_lr_pair, chs_ap_pair = ch_di_list()
    figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
    collection_dict = collect_permutation_cluster_1samp_test()
    for i_chs_pair, chs_pairs in enumerate(zip(chs_lr_pair, chs_ap_pair)):
        chs_name = ['lr', 'ap']
        fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
        colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
        colors = colors[[0,2],:] # take 0:green, 2:blue
        for i_chs, chs in enumerate(chs_pairs):
            frequency_map = load_list_pkl(file_r)
            for i in range(len(frequency_map)):
                frequency_map[i][1].pick_channels(chs)
                print(f"{frequency_map[i][1]}")
            for ((freq_name, fmin, fmax), average), ax in zip(
                    frequency_map, axes.ravel()[::-1]):
                # __import__('IPython').embed()
                # __import__('sys').exit()
                times = average.times * 1e3
                gfp = np.sum(average.data ** 2, axis=0)
                gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
                ax.plot(times, gfp, label=freq_name, color=colors[i_chs], linewidth=2.5)
                ax.axhline(0, linestyle='--', color='grey', linewidth=2)
                ci_low, ci_up = bootstrap_confidence_interval(average.data, random_state=0,
                                                              stat_fun=stat_fun)
                ci_low = rescale(ci_low, average.times, baseline=(None, 0))
                ci_up = rescale(ci_up, average.times, baseline=(None, 0))
                ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=colors[i_chs], alpha=0.3)
                if i_chs == 1:
                    T_obs, clusters, cluster_p_values, H0 = \
                        collection_dict[aging_label][task][chs_name[i_chs_pair]][(freq_name, fmin, fmax)]
                    # Create new stats image with only significant clusters
                    T_obs_plot = np.nan * np.ones_like(T_obs)
                    upper_plot = np.nan * np.ones_like(T_obs)
                    lower_plot = np.nan * np.ones_like(T_obs)
                    for c, p_val in zip(clusters, cluster_p_values):
                        if p_val <= 0.05:
                            #T_obs_plot[c] = T_obs[c]
                            upper_plot[c] = ax.get_ylim()[1]*1.1
                            lower_plot[c] = ax.get_ylim()[0]*1.1
                            print("t_obs")
                            # __import__('IPython').embed()
                            # __import__('sys').exit()
                    ax.fill_between(times, upper_plot, lower_plot, color=colors[1], alpha=0.3)
                ax.grid(True)
                ax.set_ylabel('GFP')
                ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                            xy=(0.95, 0.8),
                            horizontalalignment='right',
                            xycoords='axes fraction')
                ax.set_xlim(-100, 2000)
            axes.ravel()[-1].set_xlabel('Time [ms]')
        if aging_label:
            w_png = figures_d.joinpath(f"{aging_label}_grand_sig_clusterp_gfp_"
                                       f"{task}_{chs_name[i_chs_pair]}_dpi.png")
        else:
            w_png = figures_d.joinpath(f"grand_sig_clusterp_gfp_{task}"
                                       f"_{chs_name[i_chs_pair]}_dpi.png")
        logger.info("saving file:")
        logger.info(f"{w_png.as_posix()}")
        # collection_dict['young']['spatial']['lr'][('Theta', 4, 7)][0].shape
        fig.savefig(w_png, dpi=300)
        # __import__('IPython').embed()
        # __import__('sys').exit()


def plot_grand_sig_ttest1samp_gfp(task, file_r, aging_label=None):
    from mne.baseline import rescale
    from mne.stats import bootstrap_confidence_interval
    __import__('matplotlib').use('TkAgg')
    import matplotlib.pyplot as plt
    def stat_fun(x):
        """Return sum of squares."""
        return np.sum(x ** 2, axis=0)
    chs_lr_pair, chs_ap_pair = ch_di_list()
    figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
    collection_dict = collect_ttest_1samp()
    for i_chs_pair, chs_pairs in enumerate(zip(chs_lr_pair, chs_ap_pair)):
        chs_name = ['lr', 'ap']
        fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
        colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
        colors = colors[[0,2],:] # take 0:green, 2:blue
        for i_chs, chs in enumerate(chs_pairs):
            frequency_map = load_list_pkl(file_r)
            for i in range(len(frequency_map)):
                frequency_map[i][1].pick_channels(chs)
                print(f"{frequency_map[i][1]}")
            for ((freq_name, fmin, fmax), average), ax in zip(
                    frequency_map, axes.ravel()[::-1]):
                # __import__('IPython').embed()
                # __import__('sys').exit()
                times = average.times * 1e3
                gfp = np.sum(average.data ** 2, axis=0)
                gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
                ax.plot(times, gfp, label=freq_name, color=colors[i_chs], linewidth=2.5)
                ax.axhline(0, linestyle='--', color='grey', linewidth=2)
                ci_low, ci_up = bootstrap_confidence_interval(average.data, random_state=0,
                                                            stat_fun=stat_fun)
                ci_low = rescale(ci_low, average.times, baseline=(None, 0))
                ci_up = rescale(ci_up, average.times, baseline=(None, 0))
                ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=colors[i_chs], alpha=0.3)
                if i_chs == 1:
                    pvalue = \
                        collection_dict[aging_label][task][chs_name[i_chs_pair]][(freq_name, fmin, fmax)].pvalue
                    # Create new stats image with only significant clusters
                    #T_obs_plot = np.nan * np.ones_like(T_obs)
                    upper_plot = np.nan * np.ones_like(pvalue)
                    lower_plot = np.nan * np.ones_like(pvalue)
                    marker = np.where(pvalue < .05)
                    #T_obs_plot[c] = T_obs[c]
                    upper_plot[marker] = ax.get_ylim()[1]*1.1
                    lower_plot[marker] = ax.get_ylim()[0]*1.1
                    print("t_obs")
                    # __import__('IPython').embed()
                    # __import__('sys').exit()
                    ax.fill_between(times, upper_plot, lower_plot, color=colors[1], alpha=0.3)
                ax.grid(True)
                ax.set_ylabel('GFP')
                ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                            xy=(0.95, 0.8),
                            horizontalalignment='right',
                            xycoords='axes fraction')
                ax.set_xlim(-100, 2000)
            axes.ravel()[-1].set_xlabel('Time [ms]')
        if aging_label:
            w_png = figures_d.joinpath(f"{aging_label}_grand_sig_ttest1samp_gfp_{task}_{chs_name[i_chs_pair]}_dpi.png")
        else:
            w_png = figures_d.joinpath(f"grand_sig_gfp_{task}_{chs_name[i_chs_pair]}_dpi.png")
        logger.info("saving file:")
        logger.info(f"{w_png.as_posix()}")
        # collection_dict['young']['spatial']['lr'][('Theta', 4, 7)][0].shape
        fig.savefig(w_png, dpi=300)
        # __import__('IPython').embed()
        # __import__('sys').exit()


def plot_grand_theta_sig_ttest1samp_gfp(task_selected, aging_label=None):
    from mne.baseline import rescale
    from mne.stats import bootstrap_confidence_interval
    __import__('matplotlib').use('TkAgg')
    import matplotlib.pyplot as plt
    def stat_fun(x):
        """Return sum of squares."""
        return np.sum(x ** 2, axis=0)
    chs_lr_pair, chs_ap_pair = ch_di_list()
    figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
    collection_dict = collect_ttest_1samp()
    chs_name = ['lr', 'ap']
    chs_title_dict = {'lr': 'LH-RH', 'ap': 'ANT-POS'}
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
    count_axis = -1
    for task in task_selected:
        # if task == 'spatial':
            # continue
        file_r = f"{aging_label}_grand_{task}_frequency_map.pkl"
        for i_chs_pair, chs_pairs in enumerate(zip(chs_lr_pair, chs_ap_pair)):
            colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
            colors = colors[[0,2],:] # take 0:green, 2:blue
            count_axis += 1
            for i_chs, chs in enumerate(chs_pairs):
                frequency_map = load_list_pkl(file_r)
                for i in range(len(frequency_map)):
                    frequency_map[i][1].pick_channels(chs)
                    print(f"{frequency_map[i][1]}")
                for (freq_name, fmin, fmax), average in frequency_map:
                    # __import__('IPython').embed()
                    # __import__('sys').exit()
                    if freq_name != 'Theta':
                        print("skip")
                        continue
                    ax = axes.ravel()[::-1][count_axis]
                    times = average.times * 1e3
                    gfp = np.sum(average.data ** 2, axis=0)
                    gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
                    ax.plot(times, gfp, label=freq_name, color=colors[i_chs], linewidth=2.5)
                    ax.axhline(0, linestyle='--', color='grey', linewidth=2)
                    ci_low, ci_up = bootstrap_confidence_interval(average.data, random_state=0,
                                                                stat_fun=stat_fun)
                    ci_low = rescale(ci_low, average.times, baseline=(None, 0))
                    ci_up = rescale(ci_up, average.times, baseline=(None, 0))
                    ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=colors[i_chs], alpha=0.3)
                    if i_chs == 1:
                        pvalue = \
                            collection_dict[aging_label][task][chs_name[i_chs_pair]][(freq_name, fmin, fmax)].pvalue
                        # Create new stats image with only significant clusters
                        #T_obs_plot = np.nan * np.ones_like(T_obs)
                        upper_plot = np.nan * np.ones_like(pvalue)
                        lower_plot = np.nan * np.ones_like(pvalue)
                        marker = np.where(pvalue < .05)
                        #T_obs_plot[c] = T_obs[c]
                        upper_plot[marker] = ax.get_ylim()[1]*0.5
                        lower_plot[marker] = ax.get_ylim()[0]*0.5
                        pre_stimulus = np.where(times < 0)
                        upper_plot[pre_stimulus] = np.nan
                        lower_plot[pre_stimulus] = np.nan
                        # print("t_obs")
                        # __import__('IPython').embed()
                        # __import__('sys').exit()
                        ax.fill_between(times, upper_plot, lower_plot, color='red', alpha=0.2)
                        ax.grid(True)
                        ax.set_ylabel('GFP')
                        ax.annotate('%s: %s' % (task, chs_title_dict[chs_name[i_chs_pair]]),
                                    xy=(0.95, 0.8),
                                    horizontalalignment='right',
                                    xycoords='axes fraction')
                    ax.set_xlim(-100, 2000)
                axes.ravel()[-1].set_xlabel('Time [ms]')
    if aging_label:
        w_png = figures_d.joinpath(f"{aging_label}_grand_theta_sig_ttest1samp"
                                   f"_gfp_dpi.png")
    else:
        w_png = figures_d.joinpath(f"grand_theta_sig_ttest1samp_gfp_"
                                   f"_dpi.png")
    logger.info("saving file:")
    logger.info(f"{w_png.as_posix()}")
    # collection_dict['young']['spatial']['lr'][('Theta', 4, 7)][0].shape
    fig.savefig(w_png, dpi=300)
    # __import__('IPython').embed()
    # __import__('sys').exit()


def plot_grand_theta_sig_cluster1samp_gfp(task_selected, aging_label=None):
    from mne.baseline import rescale
    from mne.stats import bootstrap_confidence_interval
    __import__('matplotlib').use('TkAgg')
    import matplotlib.pyplot as plt
    def stat_fun(x):
        """Return sum of squares."""
        return np.sum(x ** 2, axis=0)
    chs_lr_pair, chs_ap_pair = ch_di_list()
    figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
    collection_dict = collect_permutation_cluster_1samp_test()
    chs_name = ['lr', 'ap']
    chs_title_dict = {'lr': 'LH-RH', 'ap': 'ANT-POS'}
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
    count_axis = -1
    for task in task_selected:
        # if task == 'spatial':
            # continue
        file_r = f"{aging_label}_grand_{task}_frequency_map.pkl"
        for i_chs_pair, chs_pairs in enumerate(zip(chs_lr_pair, chs_ap_pair)):
            colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
            colors = colors[[0,2],:] # take 0:green, 2:blue
            count_axis += 1
            for i_chs, chs in enumerate(chs_pairs):
                frequency_map = load_list_pkl(file_r)
                for i in range(len(frequency_map)):
                    frequency_map[i][1].pick_channels(chs)
                    print(f"{frequency_map[i][1]}")
                for (freq_name, fmin, fmax), average in frequency_map:
                    # __import__('IPython').embed()
                    # __import__('sys').exit()
                    if freq_name != 'Theta':
                        print("skip")
                        continue
                    ax = axes.ravel()[::-1][count_axis]
                    times = average.times * 1e3
                    gfp = np.sum(average.data ** 2, axis=0)
                    gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
                    ax.plot(times, gfp, label=freq_name, color=colors[i_chs], linewidth=2.5)
                    ax.axhline(0, linestyle='--', color='grey', linewidth=2)
                    ci_low, ci_up = bootstrap_confidence_interval(average.data, random_state=0,
                                                                stat_fun=stat_fun)
                    ci_low = rescale(ci_low, average.times, baseline=(None, 0))
                    ci_up = rescale(ci_up, average.times, baseline=(None, 0))
                    ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=colors[i_chs], alpha=0.3)
                    if i_chs == 1:
                        T_obs, clusters, cluster_p_values, H0 = \
                            collection_dict[aging_label][task][chs_name[i_chs_pair]][(freq_name, fmin, fmax)]
                        # Create new stats image with only significant clusters
                        T_obs_plot = np.nan * np.ones_like(T_obs)
                        upper_plot = np.nan * np.ones_like(T_obs)
                        lower_plot = np.nan * np.ones_like(T_obs)
                        pre_stimulus = np.where(times < 0)
                        for c, p_val in zip(clusters, cluster_p_values):
                            if p_val <= 0.05:
                                #T_obs_plot[c] = T_obs[c]
                                upper_plot[c] = ax.get_ylim()[1]*0.5
                                lower_plot[c] = ax.get_ylim()[0]*0.5
                        upper_plot[pre_stimulus] = np.nan
                        lower_plot[pre_stimulus] = np.nan
                        ax.fill_between(times, upper_plot, lower_plot, color='red', alpha=0.2)
                        ax.grid(True)
                        ax.set_ylabel('GFP')
                        ax.annotate('%s: %s' % (task, chs_title_dict[chs_name[i_chs_pair]]),
                                    xy=(0.95, 0.8),
                                    horizontalalignment='right',
                                    xycoords='axes fraction')
                    ax.set_xlim(-100, 2000)
                axes.ravel()[-1].set_xlabel('Time [ms]')
    if aging_label:
        w_png = figures_d.joinpath(f"{aging_label}_grand_theta_sig_cluster1samp"
                                   f"_gfp_dpi.png")
    else:
        w_png = figures_d.joinpath(f"grand_theta_sig_cluster1samp_gfp_"
                                   f"dpi.png")
    logger.info("saving file:")
    logger.info(f"{w_png.as_posix()}")
    # collection_dict['young']['spatial']['lr'][('Theta', 4, 7)][0].shape
    fig.savefig(w_png, dpi=300)
    # __import__('IPython').embed()
    # __import__('sys').exit()


def plot_grand_sig_clusterbp_gfp(task, file_r, aging_label=None):
    from mne.baseline import rescale
    from mne.stats import bootstrap_confidence_interval
    __import__('matplotlib').use('TkAgg')
    import matplotlib.pyplot as plt
    def stat_fun(x):
        """Return sum of squares."""
        return np.sum(x ** 2, axis=0)
    chs_lr_pair, chs_ap_pair = ch_di_list()
    figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
    collection_dict = collect_permutation_cluster_test()
    for i_chs_pair, chs_pairs in enumerate(zip(chs_lr_pair, chs_ap_pair)):
        chs_name = ['lr', 'ap']
        fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=True)
        colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
        colors = colors[[0,2],:] # take 0:green, 2:blue
        for i_chs, chs in enumerate(chs_pairs):
            frequency_map = load_list_pkl(file_r)
            for i in range(len(frequency_map)):
                frequency_map[i][1].pick_channels(chs)
                print(f"{frequency_map[i][1]}")
            for ((freq_name, fmin, fmax), average), ax in zip(
                    frequency_map, axes.ravel()[::-1]):
                # __import__('IPython').embed()
                # __import__('sys').exit()
                times = average.times * 1e3
                gfp = np.sum(average.data ** 2, axis=0)
                gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
                ax.plot(times, gfp, label=freq_name, color=colors[i_chs], linewidth=2.5)
                ax.axhline(0, linestyle='--', color='grey', linewidth=2)
                # ci_low, ci_up = bootstrap_confidence_interval(average.data, random_state=0,
                                                            # stat_fun=stat_fun)
                # ci_low = rescale(ci_low, average.times, baseline=(None, 0))
                # ci_up = rescale(ci_up, average.times, baseline=(None, 0))
                # ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=colors[i_chs], alpha=0.3)
                if i_chs == 1:
                    T_obs, clusters, cluster_p_values, H0 = \
                        collection_dict[aging_label][task][chs_name[i_chs_pair]][(freq_name, fmin, fmax)]
                    # Create new stats image with only significant clusters
                    T_obs_plot = np.nan * np.ones_like(T_obs)
                    upper_plot = np.nan * np.ones_like(T_obs)
                    lower_plot = np.nan * np.ones_like(T_obs)
                    for c, p_val in zip(clusters, cluster_p_values):
                        if p_val <= 0.05:
                            #T_obs_plot[c] = T_obs[c]
                            upper_plot[c] = ax.get_ylim()[1]*1.1
                            lower_plot[c] = ax.get_ylim()[0]*1.1
                            print("t_obs")
                            # __import__('IPython').embed()
                            # __import__('sys').exit()
                    ax.fill_between(times, upper_plot, lower_plot, color=colors[1], alpha=0.3)
                ax.grid(True)
                ax.set_ylabel('GFP')
                ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                            xy=(0.95, 0.8),
                            horizontalalignment='right',
                            xycoords='axes fraction')
                ax.set_xlim(-100, 2000)
            axes.ravel()[-1].set_xlabel('Time [ms]')
        if aging_label:
            w_png = figures_d.joinpath(f"{aging_label}_grand_sig_clusterbp_gfp_"
                                       f"{task}_{chs_name[i_chs_pair]}_dpi.png")
        else:
            w_png = figures_d.joinpath(f"grand_sig_clusterbp_gfp_{task}"
                                       f"_{chs_name[i_chs_pair]}_dpi.png")
        logger.info("saving file:")
        logger.info(f"{w_png.as_posix()}")
        # collection_dict['young']['spatial']['lr'][('Theta', 4, 7)][0].shape
        fig.savefig(w_png, dpi=300)
        # __import__('IPython').embed()
        # __import__('sys').exit()


def plot_di_aging_grand_gfp():
    # Plot
    task_selected = ['spatial', 'temporal']
    aging_variable = ['young', 'elder']
    for i in aging_variable:
        if i == 'elder':
            continue
        for task in task_selected:
            # if task == 'spatial':
                # continue
            file_r = f"{i}_grand_{task}_frequency_map.pkl"
            # plot_grand_gfp(task, file_r, aging_label=i)
            plot_grand_sig_clusterp_gfp(task, file_r, aging_label=i)
            # plot_grand_sig_ttest1samp_gfp(task, file_r, aging_label=i)
            # plot_grand_sig_clusterbp_gfp(task, file_r, aging_label=i)
            # plot_grand_theta_sig_ttest1samp_gfp(task, file_r, aging_label=i)


def plot_aging_grand_gfp_contrast_spatial_temporal():
    # Plot
    task_selected = ['spatial', 'temporal']
    aging_variable = ['young', 'elder']
    file_rs = list()
    for i in aging_variable:
        if i == 'elder':
            continue
        for task in task_selected:
            # if task == 'spatial':
                # continue
            file_r = f"{i}_grand_{task}_frequency_map.pkl"
            file_rs.append(file_r)
    plot_grand_gfp_contrast_spatial_temporal(file_rs, aging_label=i)


def plot_theta_aging_grand_gfp():
    # Plot
    task_selected = ['spatial', 'temporal']
    aging_variable = ['young', 'elder']
    for i in aging_variable:
        if i == 'elder':
            continue
        #plot_grand_theta_sig_ttest1samp_gfp(task_selected, aging_label=i)
        plot_grand_theta_sig_cluster1samp_gfp(task_selected, aging_label=i)


def dd_plot_gfp():
    task_selected = ['spatial', 'temporal']
    id_list, exclude_list, data_concated_dict = define_id_list()
    for task in task_selected:
        # for i in range(1):
            # sub_index = 16
        for sub_index in range(17, len(id_list)):
            logger.info(f"sub_index: {sub_index}; sub-{id_list[sub_index]}")
            logger.info(f"task: {task}")
            if str(id_list[sub_index]) in exclude_list:
                logger.info(f"excluding id:{id_list[sub_index]}")
                continue
            plot_gfp(sub_index, task)
        logger.info("Done!")
    #plot_gfp(task_selected[0])
    #plot_gfp(task_selected[1])


def main_build_individual_frequency_map_pkl():
    task_selected = ['spatial', 'temporal']
    id_list, exclude_list, data_concated_dict = define_id_list()
    for task in task_selected:
        # if task == 'spatial':
        # if task == 'temporal':
            # continue
        # for i in range(1):
            # sub_index = 24
        # for sub_index in range(0, 18):
        for sub_index in range(0, len(id_list)):
            logger.info(f"sub_index: {sub_index}; sub-{id_list[sub_index]}")
            logger.info(f"task: {task}")
            if str(id_list[sub_index]) in exclude_list:
                logger.info(f"excluding id:{id_list[sub_index]}")
                continue
            #plot_gfp(sub_index, task) #  old function
            build_individual_frequency_map_pkl(sub_index, task)
        logger.info("Done!")
    #plot_gfp(task_selected[0])
    #plot_gfp(task_selected[1])


def sample_channels():
    raw = devp_read_raw_bids()
    sfreq = raw.info['sfreq']
    data, times = raw[:5, int(sfreq * 1):int(sfreq * 3)]
    plt.plot(times, data.T)
    plt.title('Sample channels')
    plt.savefig("viz_sample_channels.png")


def sample_channels_type():
    import matplotlib as mpl
    mpl.use('TkAgg')
    raw = devp_read_raw_bids()
    sfreq = raw.info['sfreq']
    #data, times = raw[:5, int(sfreq * 1):int(sfreq * 3)]
    # need to load data for the following process
    raw = raw.load_data()
    meg_only = raw.copy().pick_types(meg=True)
    eeg_only = raw.copy().pick_types(meg=False, eeg=True)
    # no grad channel name
    #grad_only = raw.copy().pick_types(meg='grad')
    # __import__('IPython').embed()
    # __import__('sys').exit()
    #data = eeg_only.get_data()
    #eeg_only.plot();
    #meg_only.plot_psd();
    #meg_only.plot();
    raw.plot()
    # fig, ax = plt.subplots(1, 1)
    # meg_only.plot_psd(spatial_colors=False, show=False, ax=ax);
    # for freq in [60., 120., 180.]:
        # ax.axvline(freq, linestyle='--', alpha=0.6)
    # fig.savefig("viz_meg_psd.png")
    # plt.plot(raw.times, data.T)
    # plt.title('Sample channels')
    # plt.savefig("viz_eeg_only.png")


def old():
    #devp_ssp_em()
    #preprocess_pipeline()
    #devp_read_raw_bids()
    #read_eprime_csv()
    #read_eprime_csv(sub_index=45)
    #plot_events_id()
    #plot_time()
    #sample_channels()
    #sample_channels_type()
    #read_csv2()
    #read_csv()
    #read_csv3()
    #correct_time_head_filter()
    #read_lines()
    #head_filter()
    #read_con()
    #define_id_list()
    #build_id_task_dict()
    #epochs_filter_by_trial()
    #devp_gfp()
    #devp_plot_layout()
    #devp_concatenate_raw()
    #devp_plot_grand_gfp()
    # concatenate_raw_sub228()
    return None


if __name__ == '__main__':
    # concatenate_raw_sub126()
    # main_build_individual_frequency_map_pkl()
    # build_grand_frequency_maps_pkl()
    # build_aging_grand_frequency_maps_pkl()
    # plot_di_aging_grand_gfp()
    # plot_theta_aging_grand_gfp()
    # build_individual_aging_frequency_gfp()
    # collect_permutation_cluster_1samp_test()
    # collect_ttest_1samp()
    # build_individual_aging_frequency_gfp_dict()
    # save_individual_aging_frequency_gfp_dict(sub_index=None, task_selected=None)
    plot_aging_grand_gfp_contrast_spatial_temporal()
