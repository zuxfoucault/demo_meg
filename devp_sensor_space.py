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

    def get_grand_event_evoked_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            f'grand_evoked_{kwargs["task"]}'
            f'_id-{kwargs["target_event"]}.pkl').as_posix()

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

    def get_grand_event_evoked(self, subjects, task, target_event):
        """Get grand evoked by task and target_event"""
        import copy
        from devp_basicio import load_pkl, save_pkl
        kwargs = dict(task=task, target_event=target_event)
        grand_event_evoked_fname = self.get_grand_event_evoked_fname(**kwargs)

        if Path(grand_event_evoked_fname).exists():
            self.logger.info(f'Found {grand_event_evoked_fname}')
            grand_evoked = load_pkl(grand_event_evoked_fname, logger=self.logger)
            return grand_evoked

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
        save_pkl(grand_evoked, grand_event_evoked_fname, logger=self.logger)
        return grand_evoked


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
    for task in ss.task_list:
        grand_evoked = \
            ss.get_grand_event_evoked(subjects, task, target_event)


def main():
    main_sensor_space_plot()


if __name__ == '__main__':
    main()
