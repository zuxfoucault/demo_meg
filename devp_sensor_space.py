import os
import loggy
from pathlib import Path
import mne
import numpy as np


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


class Logger(object):
    def __init__(self):
        self.info = print


class SensorSpace(object):
    """Sensor space related operation"""
    def __init__(self):
        self.logger = logger
        self.SUBJECTS_DIR = Path(
            '/home/foucault/data/derivatives/working_memory/freesurfer')
        self.mne_derivatives_dname = Path(
            f'/home/foucault/data/derivatives/working_memory/mne')
        self.intermediate_dname = Path(
            f'/home/foucault/data/derivatives/working_memory/intermediate')
        self.task_list = ['spatial', 'temporal']

    def get_event_epochs_config(self):
        tmin, tmax = -0.5, 2
        tmin_plot, tmax_plot = -0.1, 1.75
        return tmin, tmax, tmin_plot, tmax_plot

    def get_grand_event_evoked_pkl(self, **kwargs):
        return self.intermediate_dname.joinpath(
            f'grand_evoked_{kwargs["task"]}'
            f'_id-{kwargs["target_event"]}.pkl').as_posix()

    def get_grand_event_evoked_png(self, **kwargs):
        return self.intermediate_dname.joinpath(
            'figure',
            f'grand_evoked_{kwargs["task"]}'
            f'_id-{kwargs["target_event"]}.png').as_posix()

    def get_grand_event_evoked_contrast_png(self, **kwargs):
        return self.intermediate_dname.joinpath(
            'figure',
            f'grand_evoked_contrast'
            f'_id-{kwargs["target_event"]}.png').as_posix()

    def get_event_epochs(self, subject, task, target_event):
        """Get epochs time lock to encoder stimulus target_event"""
        from devp_source_space import reconstruct_raw_w_file
        self.logger.info(f'Get epochs time lock to encoder stimulus {target_event}...')
        recon_raw, events, filt_raw_em = reconstruct_raw_w_file(
            subject, task, base_file=self.mne_derivatives_dname,
            logger=self.logger)
        event_ids = list(np.unique(events[:,2]))
        # fine-tune time length
        tmin, tmax, tmin_plot, tmax_plot = self.get_event_epochs_config()
        baseline = None
        logger.info(f"target event: {target_event}")
        # remove the reject=dict(mag=1.5e-12) option, too much data point lost
        epochs = mne.Epochs(recon_raw, events, target_event, tmin, tmax,
                            baseline=baseline, picks='meg',
                            preload=False)
        return epochs, recon_raw, events

    def get_event_evoked(self, subject, task, target_event):
        """Get evoked time lock to encoder stimulus target_event"""
        epochs, recon_raw, events = \
            self.get_event_epochs(subject, task, target_event)
        evoked = epochs.average()
        return evoked

    def get_grand_event_evoked(self, subjects, **kwargs):
        """Get grand evoked by task and target_event"""
        import copy
        from devp_basicio import load_pkl, save_pkl
        grand_event_evoked_pkl = self.get_grand_event_evoked_pkl(**kwargs)

        if Path(grand_event_evoked_pkl).exists():
            self.logger.info(f'Found {grand_event_evoked_pkl}')
            grand_evoked = load_pkl(grand_event_evoked_pkl, logger=self.logger)
            return grand_evoked

        task = kwargs['task']
        target_event = kwargs['target_event']
        count = 0
        for subject in subjects:
            self.logger.info(f'Processing {subject}')
            evoked = \
                self.get_event_evoked(subject, task, target_event)

            count += 1
            if count == 1:
                grand_evoked = copy.deepcopy(evoked)
                # prevent data is nan in first item
                if np.isnan(evoked.data).any():
                    self.logger.info(f'Data contains NaN')
                    grand_evoked.data = \
                        np.zeros(evoked.data.shape)
                    grand_evoked.nave = 0
                else:
                    # reset nave to n=1 for grand average
                    grand_evoked.nave = 1
                continue

            if np.isnan(evoked.data).any():
                self.logger.info(f'Data contains NaN')
                continue

            grand_evoked.data += evoked.data
            grand_evoked.nave += 1

        # get average
        grand_evoked.data /= grand_evoked.nave
        logger.info(f"#participants in task_{task}: {count}")
        logger.info(f"#data contained nan: {len(subjects)-grand_evoked.nave}")
        save_pkl(grand_evoked, grand_event_evoked_pkl, logger=self.logger)
        return grand_evoked

    def get_event_evoked_list(self, subjects, **kwargs):
        """Get grand evoked by task and target_event"""
        import copy
        from devp_basicio import load_pkl, save_pkl
        task = kwargs['task']
        target_event = kwargs['target_event']
        count = 0
        _list = list()
        for subject in subjects:
            self.logger.info(f'Processing {subject}')
            evoked = \
                self.get_event_evoked(subject, task, target_event)
            _list.append(evoked)
        return _list

    def get_tfr_mort_power(self, _evoked):
        # define frequencies of interest (log-spaced)
        freqs = np.logspace(*np.log10([6, 35]), num=8)
        n_cycles = freqs / 2.  # different number of cycle per frequency
        power = mne.time_frequency.tfr_morlet(
            _evoked, freqs=freqs, n_cycles=n_cycles, use_fft=True,
            return_itc=False, decim=3, n_jobs=1)
        return power

    def plot_tfr_mort(self, grand_evoked, **kwargs):
        power = self.get_tfr_mort_power(grand_evoked)
        # return_itc must be False for evoked data
        fig = power.plot_topo(
            baseline=(-0.5, 0), mode='logratio', title='Average power',
            show=False)
        grand_event_evoked_png = \
            self.get_grand_event_evoked_png(**kwargs)
        self.logger.info(f'Saving {grand_event_evoked_png}')
        fig.savefig(grand_event_evoked_png, dpi=300)

    def plot_tfr_mort_contrast(self, subjects, **kwargs):
        import matplotlib.pyplot as plt
        power_list = list()
        for task in self.task_list:
            kwargs['task'] = task
            grand_evoked = \
                self.get_grand_event_evoked(subjects, **kwargs)
            _power = self.get_tfr_mort_power(grand_evoked)
            power_list.append(_power)
        # return_itc must be False for evoked data
        power_list[0].data = power_list[0].data - power_list[1].data
        vmax = power_list[0].data.max()
        vmin = power_list[0].data.min()
        #fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharey=True)
        fig = power_list[0].plot_topo(
            baseline=(-0.5, 0), mode='mean', title='Time-frequescy:\n Spatial - temporal WM contrast',
            show=False, font_color='k')
        # fig = power_list[0].plot_topo(
            # baseline=(-0.5, 0), mode='mean', title='Spatial - temporal WM contrast',
            # show=False, vmin=vmin, vmax=vmax)
        # fig = power_list[0].plot_topo(
            # baseline=(-0.5, 0), mode='logratio', title='Average power',
            # show=False, vmin=vmin, vmax=vmax)
        grand_event_evoked_contrast_png = \
            self.get_grand_event_evoked_contrast_png(**kwargs)
        self.logger.info(f'Saving {grand_event_evoked_contrast_png}')
        # __import__('IPython').embed()
        # __import__('sys').exit()
        fig.savefig(grand_event_evoked_contrast_png, dpi=300, pad_inches=0.5)


class SensorSpace2(SensorSpace):
    """Target event: maintainance; id: 239
    Target event: retrieval; id: 223
    event duration: 4"""
    def __init__(self):
        super().__init__()
        self.logger = logger

    def get_event_epochs_config(self):
        tmin, tmax = -0.5, 4
        tmin_plot, tmax_plot = -0.1, 1.75
        return tmin, tmax, tmin_plot, tmax_plot


def main_sensor_space_plot():
    from devp_basicio import define_id_by_aging_dict
    from devp_source_space import get_subjects_list
    ss = SensorSpace()
    ss.logger = logger
    subjects = get_subjects_list()
    aging_dict = define_id_by_aging_dict()
    _subjects = list()
    for subject in subjects:
        # pick just young
        if int(subject[-3:]) not in aging_dict['young']:
            logger.info(f'Filtering out {subject} ...')
            continue
        _subjects.append(subject)
        logger.info(f'Appending {subject} ...')
    subjects = _subjects
    target_event = 253
    kwargs = dict(task=None, target_event=target_event)
    # for task in ss.task_list:
        # kwargs['task'] = task
        # grand_evoked = \
            # ss.get_grand_event_evoked(subjects, **kwargs)
        # ss.plot_tfr_mort(grand_evoked, **kwargs)
        # evoked_list = \
            # ss.get_event_evoked_list(subjects, **kwargs)
        # ss.plot_tfr_mort(evoked_list, **kwargs)
    ss.plot_tfr_mort_contrast(subjects, **kwargs)


def main_sensor_space_plot2():
    """Target event: maintenance; id: 239"""
    from devp_basicio import define_id_by_aging_dict
    from devp_source_space import get_subjects_list
    ss = SensorSpace2()
    ss.logger = logger
    subjects = get_subjects_list()
    aging_dict = define_id_by_aging_dict()
    _subjects = list()
    for subject in subjects:
        # pick just young
        if int(subject[-3:]) not in aging_dict['young']:
            logger.info(f'Filtering out {subject} ...')
            continue
        _subjects.append(subject)
        logger.info(f'Appending {subject} ...')
    subjects = _subjects
    target_event = 239
    kwargs = dict(task=None, target_event=target_event)
    ss.plot_tfr_mort_contrast(subjects, **kwargs)


def main_sensor_space_plot3():
    """Target event: retrieval; id: 223"""
    from devp_basicio import define_id_by_aging_dict
    from devp_source_space import get_subjects_list
    ss = SensorSpace2()
    ss.logger = logger
    subjects = get_subjects_list()
    aging_dict = define_id_by_aging_dict()
    _subjects = list()
    for subject in subjects:
        # pick just young
        if int(subject[-3:]) not in aging_dict['young']:
            logger.info(f'Filtering out {subject} ...')
            continue
        _subjects.append(subject)
        logger.info(f'Appending {subject} ...')
    subjects = _subjects
    target_event = 223
    kwargs = dict(task=None, target_event=target_event)
    ss.plot_tfr_mort_contrast(subjects, **kwargs)


def main():
    main_sensor_space_plot3()


if __name__ == '__main__':
    main()
