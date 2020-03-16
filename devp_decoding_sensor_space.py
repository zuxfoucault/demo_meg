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


class DecodingSensorSpace(object):
    """Sensor space related operation"""
    def __init__(self):
        self.logger = logger
        self.SUBJECTS_DIR = Path(
            '/home/foucault/data/derivatives/working_memory/freesurfer')
        self.mne_derivatives_d = Path(
            f'/home/foucault/data/derivatives/working_memory/mne')
        self.intermediate_d = Path(
            f'/home/foucault/data/derivatives/working_memory/intermediate')
        self.figures_d = Path(
            "/home/foucault/data/derivatives/working_memory/intermediate/figure")
        self.task_list = ['spatial', 'temporal']
        self.task_label_dict = dict(spatial=0, temporal=1)

    def get_plot_raw_png(self, **kwargs):
        return self.figures_d.joinpath(
            f'raw_{kwargs["name"]}.png').as_posix()

    def get_decoding_model_pkl(self, **kwargs):
        return self.intermediate_d.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}_'
            f'decoding_{kwargs["name"]}.pkl').as_posix()

    def get_decoding_score_pkl(self, **kwargs):
        return self.intermediate_d.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}_'
            f'decoding_{kwargs["name"]}.pkl').as_posix()

    def check_subject_intermediate_directory(self, **kwargs):
        d = self.intermediate_d.joinpath(kwargs['subject'])
        if not d.exists():
            self.logger.info(f'Make directory: {d}')
            d.mkdir(parents=True, exist_ok=True)

    def get_event_epochs_config(self):
        tmin, tmax = -0.2, 2
        tmin_plot, tmax_plot = -0.1, 1.75
        return tmin, tmax, tmin_plot, tmax_plot

    def get_event_epochs(self, subject, task, target_event):
        """Get epochs time lock to encoder stimulus target_event"""
        from devp_source_space import reconstruct_raw_w_file
        self.logger.info(f'Get epochs time lock to encoder stimulus {target_event}...')
        recon_raw, events, filt_raw_em = reconstruct_raw_w_file(
            subject, task, base_file=self.mne_derivatives_d,
            logger=self.logger)
        # event_ids = list(np.unique(events[:,2]))
        # fine-tune time length
        tmin, tmax, tmin_plot, tmax_plot = self.get_event_epochs_config()
        baseline = None
        logger.info(f"target event: {target_event}")
        # remove the reject=dict(mag=1.5e-12) option, too much data point lost
        epochs = mne.Epochs(recon_raw, events, target_event, tmin, tmax,
                            baseline=baseline, picks='meg',
                            preload=False)
        return epochs, recon_raw, events

    def get_event_epochs_resampled(self, subject, task, target_event, new_sampling_rate):
        """Get epochs time lock to encoder stimulus target_event
        Resampling"""
        from devp_source_space import reconstruct_raw_w_file
        self.logger.info(f'Get epochs time lock to encoder stimulus {target_event}...')
        recon_raw, events, filt_raw_em = reconstruct_raw_w_file(
            subject, task, base_file=self.mne_derivatives_d,
            logger=self.logger)
        recon_raw.load_data()
        self.logger.info(f"Original sampling rate: {recon_raw.info['sfreq']} Hz")
        raw_resampled, events = recon_raw.copy().resample(
            sfreq=new_sampling_rate,  events=events)
        self.logger.info(f"New sampling rate: {raw_resampled.info['sfreq']} Hz")
        # event_ids = list(np.unique(events[:,2]))
        # fine-tune time length
        tmin, tmax, tmin_plot, tmax_plot = self.get_event_epochs_config()
        baseline = (None, 0)
        logger.info(f"target event: {target_event}")
        # remove the reject=dict(mag=1.5e-12) option, too much data point lost
        epochs = mne.Epochs(raw_resampled, events, target_event, tmin, tmax,
                            baseline=baseline, picks='meg', reject=dict(mag=1.5e-10),
                            preload=False)
        return epochs, raw_resampled, events

    def devp_estimator_gat(self, **kwargs):
        from mne.decoding import GeneralizingEstimator, LinearModel
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        from sklearn.metrics import make_scorer
        from sklearn.model_selection import StratifiedKFold
        from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                                  Scaler, cross_val_multiscore, LinearModel,
                                  get_coef, Vectorizer, CSP)
        from jr.gat import scorer_spearman
        clf = make_pipeline(StandardScaler(), LinearModel(Ridge()))
        scorer = scorer_spearman
        kwargs = dict()
        gat = GeneralizingEstimator(clf, scoring=make_scorer(scorer),
                                    n_jobs=6, **kwargs)
        return gat

    def devp_get_data_sensor_space_gat(self, **kwargs):
        """Matrix size need to determine"""
        subject = kwargs['subject']
        target_event = kwargs['target_event']
        new_sampling_rate = 120
        X_list = list()
        lable_list = list()
        lable_index = [0, 1]  # spatial and temporal
        for i, i_task in enumerate(self.task_list):
            epochs, raw_resampled, events =\
                self.get_event_epochs_resampled(
                    subject, i_task, target_event, new_sampling_rate)
            # X_list.append(epochs.get_data())
            X_list.extend(epochs.get_data())
            y = [lable_index[i]]*epochs.get_data().shape[0]
            lable_list.extend(y)
        y = np.array(lable_list, dtype=float)
        X = np.array(X_list, dtype=float)
        #y = np.broadcast_to(y, (X.shape[2],len(y)))
        return X, y

    def devp_estimator(self, **kwargs):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                                  Scaler, cross_val_multiscore, LinearModel,
                                  get_coef, Vectorizer, CSP)
        from sklearn.metrics import make_scorer
        clf = make_pipeline(StandardScaler(),
                            LinearModel(LogisticRegression(solver='lbfgs')))
        time_decod = SlidingEstimator(clf, n_jobs=6,
                                      scoring='roc_auc',
                                      verbose=True)
        return time_decod

    def devp_get_data_sensor_space(self, **kwargs):
        subject = kwargs['subject']
        target_event = kwargs['target_event']
        new_sampling_rate = 120
        X_list = list()
        lable_list = list()
        lable_index = [2, 3]  # spatial and temporal
        for i, i_task in enumerate(self.task_list):
            epochs, raw_resampled, events =\
                self.get_event_epochs_resampled(
                    subject, i_task, target_event, new_sampling_rate)
            # X_list.append(epochs.get_data())
            X_list.extend(epochs.get_data())
            y = [lable_index[i]]*epochs.get_data().shape[0]
            lable_list.extend(y)
        y = np.array(lable_list, dtype=float)
        X = np.array(X_list, dtype=float)
        return X, y, epochs

    def plot_auc(self, epochs, scores, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(epochs.times, scores, label='score')
        ax.axhline(.5, color='k', linestyle='--', label='chance')
        ax.set_xlabel('Times')
        ax.set_ylabel('AUC')  # Area Under the Curve
        ax.legend()
        ax.axvline(.0, color='k', linestyle='-')
        ax.set_title(f'Sensor space decoding: {kwargs["target_event"]}')
        kwargs.update(dict(name=f'rocauc_{kwargs["target_event"]}'))
        raw_png = self.get_plot_raw_png(**kwargs)
        self.logger.info(f'Saving: {raw_png}')
        fig.savefig(raw_png, dpi=300)

    def plot_decoding_joint(self, coef, epochs, **kwargs):
        evoked = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
        joint_kwargs = dict(ts_args=dict(time_unit='s'),
                            topomap_args=dict(time_unit='s'))
        fig = evoked.plot_joint(times=np.arange(0., 2.00, .500),
                                title='patterns', show=False,
                                **joint_kwargs)
        kwargs.update(dict(name=f'decoding_joint_{kwargs["target_event"]}'))
        raw_png = self.get_plot_raw_png(**kwargs)
        self.logger.info(f'Saving: {raw_png}')
        fig.savefig(raw_png, dpi=300)

    def devp_fit_decoding_sensor_space(self, **kwargs):
        from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                                  Scaler, cross_val_multiscore, LinearModel,
                                  get_coef, Vectorizer, CSP)
        from devp_basicio import save_pkl, load_pkl
        kwargs.update(dict(name=f'model_{kwargs["target_event"]}'))
        model_f = self.get_decoding_model_pkl(**kwargs)
        kwargs.update(dict(name=f'score_{kwargs["target_event"]}'))
        scores_f = self.get_decoding_score_pkl(**kwargs)
        if Path(model_f).exists() and Path(scores_f).exists():
            time_decod = load_pkl(model_f, logger=self.logger)
            scores = load_pkl(scores_f, logger=self.logger)
            if kwargs['retrain'] == False:
                return scores, None
        X, y, epochs = self.devp_get_data_sensor_space(**kwargs)
        time_decod = self.devp_estimator(**kwargs)
        time_decod.fit(X, y)
        save_pkl(time_decod, model_f, logger=self.logger)
        scores = cross_val_multiscore(time_decod, X, y, cv=12, n_jobs=6)
        scores = np.mean(scores, axis=0)
        save_pkl(scores, scores_f, logger=self.logger)
        # self.plot_auc(epochs, scores, **kwargs)
        # coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
        # self.plot_decoding_joint(coef, epochs, **kwargs)
        return scores, epochs

    def devp_decoding_sensor_space(self, **kwargs):
        """Get decoding grand AUC"""
        scores_list = list()
        for subject in kwargs['subjects']:
            kwargs.update(dict(subject=subject))
            scores, epochs = \
                self.devp_fit_decoding_sensor_space(**kwargs)
            scores_list.append(scores)
        self.logger.info(f'Len(list): {len(scores_list)}')
        scores = np.array(scores_list)
        self.logger.info(f'Shape(scores): {scores.shape}')
        scores = np.mean(scores, axis=0)
        X, y, epochs = self.devp_get_data_sensor_space(**kwargs)
        self.plot_auc(epochs, scores, **kwargs)


def main_viz_resample_raw():
    """Visualize the resampling results"""
    from devp_basicio import define_id_by_aging_dict
    from devp_source_space import get_subjects_list, _read_raw_bids
    ss = DecodingSensorSpace()
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
    target_event = 247
    kwargs = dict(task=None, target_event=target_event)
    # epochs, recon_raw, events =\
        # ss.get_event_epochs(subjects[0], ss.task_list[0], target_event)
    epochs, raw_resampled, events =\
        ss.get_event_epochs_resampled(
            subjects[0], ss.task_list[0], target_event, 250)
    # raw, events, event_id, raw_em = _read_raw_bids(subjects[0], ss.task_list[0])
    raw_resampled.load_data()
    fig = raw_resampled.copy().pick_types(meg=True, stim=True).plot(events=events, start=50, duration=200)
    kwargs = dict(name='evnet_align')
    raw_png = ss.get_plot_raw_png(**kwargs)
    ss.logger.info(f'Saving: {raw_png}')
    fig.savefig(raw_png)


def main_decoding_compute_recon():
    from devp_basicio import define_id_by_aging_dict
    from devp_source_space import (get_subjects_list, _read_raw_bids,
                                   get_event_id_dict)
    ss = DecodingSensorSpace()
    ss.logger = logger
    subjects = get_subjects_list()
    aging_dict = define_id_by_aging_dict()
    event_id_dict = get_event_id_dict()
    _subjects = list()
    for subject in subjects:
        # pick just young
        if int(subject[-3:]) not in aging_dict['young']:
            logger.info(f'Filtering out {subject} ...')
            continue
        if int(subject[-3:]) in [124, 125]:
            continue # mag reject
        _subjects.append(subject)
        logger.info(f'Appending {subject} ...')
    subjects = _subjects
    kwargs = dict(subjects=subjects, target_event=event_id_dict['maintenance'],
                  retrain=False)
    ss.devp_decoding_sensor_space(**kwargs)
    # for subject in subjects:
        # kwargs = dict(subject=subject, target_event=253)
        # ss.check_subject_intermediate_directory(**kwargs)
    #target_event_list = []
    # target_event = 247
    # kwargs = dict(subject=subjects[0], target_event=253)
    # ss.devp_decoding_sensor_space(**kwargs)


def main():
    main_decoding_compute_recon()


if __name__ == '__main__':
    main()
