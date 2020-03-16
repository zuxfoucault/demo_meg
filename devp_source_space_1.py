import os
import loggy
from pathlib import Path
import mne
import numpy as np
from devp_source_space import SourceSpaceStat4


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


class SourceSpaceStat5(SourceSpaceStat4):
    """5D time-frequency beamforming based on LCMV
    with covariance from sti1 baseline
    """
    def __init__(self):
        super().__init__()
        self.logger = logger

    def get_lcmv_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}_{kwargs["task_selected"]}'
            f'_{kwargs["target_event_id"]}_covariance-baseline_lcmv.pkl').as_posix()

    def get_lcmv_f4_f45_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}_{kwargs["task_selected"]}'
            f'_{kwargs["target_event_id"]}_covariance-baseline_f4_f45_lcmv.pkl').as_posix()

    def get_lcmv_f4_f45_spatial_temporal_contrast_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}'
            f'_{kwargs["target_event_id"]}_covariance-baseline_f4_f45_lcmv_contrast.pkl').as_posix()

    def get_lcmv_spatio_temporal_cluster_1samp_test_fname(self, **kwargs):
        return f'lcmv_{kwargs["freq"]}_{kwargs["target_event"]}_cov-baseline_spatio_temporal_cluster_1samp_test.pkl'

    def get_plot_lcmv_freq_f4_f45_clusters_fname(self, **kwargs):
        figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
        return figures_d.joinpath(
            f'lcmv_{kwargs["freq"]}_{kwargs["target_event_id"]}_'
            'f4_f45_covariance_baseline_clusters.png').as_posix()

    def get_lcmv_event_epochs_encoder(self, subject, task_selected, target_event):
        """Get epochs time lock to encoder stimulus target_event"""
        from devp_source_space import reconstruct_raw_w_file
        self.logger.info(f'Get epochs time lock to encoder stimulus {target_event}...')
        # recon_raw, events, filt_raw_em = reconstruct_raw(subject, task_selected)
        # new version
        recon_raw, events, filt_raw_em = reconstruct_raw_w_file(
            subject, task_selected, base_file=self.mne_derivatives_dname,
            logger=self.logger)
        # event_ids = list(np.unique(events[:,2]))
        # fine-tune time length
        tmin, tmax, tmin_plot, tmax_plot = self.get_lcmv_event_epochs_config()
        baseline = None
        logger.info(f"target event: {target_event}")
        # remove the reject=dict(mag=1.5e-12) option, too much data point lost
        epochs = mne.Epochs(recon_raw, events, target_event, tmin, tmax,
                            baseline=baseline, picks='meg',
                            preload=False)
        return epochs, recon_raw, events

    def get_lcmv_event_epochs_encoder_resampled(
        self, subject, task_selected, target_event, new_sampling_rate):
        """Get epochs time lock to encoder stimulus target_event
        Resampling"""
        from devp_source_space import reconstruct_raw_w_file
        self.logger.info(f'Get epochs time lock to encoder stimulus {target_event}...')
        # recon_raw, events, filt_raw_em = reconstruct_raw(subject, task_selected)
        # new version
        recon_raw, events, filt_raw_em = reconstruct_raw_w_file(
            subject, task_selected, base_file=self.mne_derivatives_dname,
            logger=self.logger)
        recon_raw.load_data()
        self.logger.info(f"Original sampling rate:{recon_raw.info['sfreq']} Hz")
        raw_resampled, events = recon_raw.copy().resample(sfreq=500,  events=events)
        self.logger.info(f"New sampling rate:{raw_resampled.info['sfreq']} Hz")
        event_ids = list(np.unique(events[:,2]))
        # fine-tune time length
        tmin, tmax, tmin_plot, tmax_plot = self.get_lcmv_event_epochs_config()
        baseline = None
        logger.info(f"target event: {target_event}")
        # remove the reject=dict(mag=1.5e-12) option, too much data point lost
        #epochs = mne.Epochs(recon_raw, events, target_event, tmin, tmax,
        epochs = mne.Epochs(raw_resampled, events, target_event, tmin, tmax,
                            baseline=baseline, picks='meg',
                            preload=False)
        return epochs, raw_resampled, events

    def get_lcmv_solution(self, subject, task_selected, target_event_id):
        # Make sure the number of noise epochs is the same as data epochs
        tmin, tmax, tmin_plot, tmax_plot = self.get_lcmv_event_epochs_config()
        epochs, raw, events = \
            self.get_lcmv_event_epochs_encoder(subject,
                                               task_selected,
                                               target_event_id)
        fwd_fname = self.get_fwd_fname(subject=subject)
        forward = mne.read_forward_solution(fwd_fname)
        # Setting frequency bins as in Dalal et al. 2008 (high gamma was subdivided)
        # freq_bins = [(4, 12), (12, 30), (30, 55), (65, 299)]  # Hz
        # corresponding to get_lcmv_fname
        iter_freqs = [
            ('Theta', 4, 7),
            ('Alpha', 8, 12),
            ('Beta', 13, 25),
            ('Gamma', 30, 45)
        ]
        freq_bins = [(i[1], i[2]) for i in iter_freqs]
        win_lengths = [0.3, 0.2, 0.15, 0.1]  # s
        # Setting the time step
        tstep = 0.05
        # Setting the whitened data covariance regularization parameter
        data_reg = 0.001
        # Subtract evoked response prior to computation?
        subtract_evoked = False
        # Calculating covariance from empty room noise. To use baseline data as noise
        # substitute raw for raw_noise, epochs.events for epochs_noise.events, tmin for
        # desired baseline length, and 0 for tmax_plot.
        # Note, if using baseline data, the averaged evoked response in the baseline
        # period should be flat.
        noise_covs = []
        event_id = 254  # use calibrated with first stimuli
        for (l_freq, h_freq) in freq_bins:
            raw_band = raw.copy()
            raw_band.filter(l_freq, h_freq, n_jobs=1, fir_design='firwin')
            # __import__('IPython').embed()
            # __import__('sys').exit()
            epochs_band = mne.Epochs(raw_band, events, event_id,
                                     tmin=tmin_plot, tmax=0, baseline=None,
                                     proj=True, picks='meg')
            noise_cov = mne.compute_covariance(epochs_band, method='auto', rank=None)
            noise_covs.append(noise_cov)
            del raw_band  # to save memory
        # Computing LCMV solutions for time-frequency windows in a label in source
        # space for faster computation, use label=None for full solution
        stcs = mne.beamformer.tf_lcmv(
            epochs, forward, noise_covs, tmin, tmax, tstep, win_lengths,
            freq_bins=freq_bins, subtract_evoked=subtract_evoked,
            reg=data_reg, rank=None)
        return stcs


class SourceSpaceStat6(SourceSpaceStat5):
    """5D time-frequency beamforming based on LCMV
    with covariance from sti1 baseline
    Substract from evoked when beamforming
    """
    def __init__(self):
        super().__init__()
        self.logger = logger

    def get_lcmv_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}_{kwargs["task_selected"]}'
            f'_{kwargs["target_event_id"]}_covariance-baseline_wo_evoked_lcmv.pkl').as_posix()

    def get_lcmv_f4_f45_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}_{kwargs["task_selected"]}'
            f'_{kwargs["target_event_id"]}_covariance-baseline_wo_evoked_f4_f45_lcmv.pkl').as_posix()

    def get_lcmv_f4_f45_spatial_temporal_contrast_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}'
            f'_{kwargs["target_event_id"]}_covariance-baseline_wo_evoked_f4_f45_lcmv_contrast.pkl').as_posix()

    def get_lcmv_spatio_temporal_cluster_1samp_test_fname(self, **kwargs):
        return f'lcmv_{kwargs["freq"]}_{kwargs["target_event"]}_cov-baseline_wo_evoked_spatio_temporal_cluster_1samp_test.pkl'

    def get_plot_lcmv_freq_f4_f45_clusters_fname(self, **kwargs):
        figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
        return figures_d.joinpath(
            f'lcmv_{kwargs["freq"]}_{kwargs["target_event_id"]}_'
            'f4_f45_covariance_baseline_wo_evoked_clusters.png').as_posix()

    def get_lcmv_solution(self, subject, task_selected, target_event_id):
        # Make sure the number of noise epochs is the same as data epochs
        tmin, tmax, tmin_plot, tmax_plot = self.get_lcmv_event_epochs_config()
        epochs, raw, events = \
            self.get_lcmv_event_epochs_encoder(subject,
                                               task_selected,
                                               target_event_id)
        fwd_fname = self.get_fwd_fname(subject=subject)
        forward = mne.read_forward_solution(fwd_fname)
        # Setting frequency bins as in Dalal et al. 2008 (high gamma was subdivided)
        # freq_bins = [(4, 12), (12, 30), (30, 55), (65, 299)]  # Hz
        # corresponding to get_lcmv_fname
        iter_freqs = [
            ('Theta', 4, 7),
            ('Alpha', 8, 12),
            ('Beta', 13, 25),
            ('Gamma', 30, 45)
        ]
        freq_bins = [(i[1], i[2]) for i in iter_freqs]
        win_lengths = [0.3, 0.2, 0.15, 0.1]  # s
        # Setting the time step
        tstep = 0.05
        # Setting the whitened data covariance regularization parameter
        data_reg = 0.001
        # Subtract evoked response prior to computation?
        subtract_evoked = True  # modified from previous version
        # Calculating covariance from empty room noise. To use baseline data as noise
        # substitute raw for raw_noise, epochs.events for epochs_noise.events, tmin for
        # desired baseline length, and 0 for tmax_plot.
        # Note, if using baseline data, the averaged evoked response in the baseline
        # period should be flat.
        noise_covs = []
        event_id = 254  # use calibrated with first stimuli
        for (l_freq, h_freq) in freq_bins:
            raw_band = raw.copy()
            raw_band.filter(l_freq, h_freq, n_jobs=1, fir_design='firwin')
            # __import__('IPython').embed()
            # __import__('sys').exit()
            epochs_band = mne.Epochs(raw_band, events, event_id,
                                     tmin=tmin_plot, tmax=0, baseline=None,
                                     proj=True, picks='meg')
            noise_cov = mne.compute_covariance(epochs_band, method='auto', rank=None)
            noise_covs.append(noise_cov)
            del raw_band  # to save memory
        # Computing LCMV solutions for time-frequency windows in a label in source
        # space for faster computation, use label=None for full solution
        stcs = mne.beamformer.tf_lcmv(
            epochs, forward, noise_covs, tmin, tmax, tstep, win_lengths,
            freq_bins=freq_bins, subtract_evoked=subtract_evoked,
            reg=data_reg, rank=None)
        return stcs


class SourceSpaceStat7(SourceSpaceStat5):
    """5D time-frequency beamforming based on LCMV
    with covariance from sti1 baseline
    Downsample for TFCE
    """
    def __init__(self):
        super().__init__()
        self.logger = logger

    def get_lcmv_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}_{kwargs["task_selected"]}'
            f'_{kwargs["target_event_id"]}_covariance-baseline_downsample_lcmv.pkl').as_posix()

    def get_lcmv_f4_f45_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}_{kwargs["task_selected"]}'
            f'_{kwargs["target_event_id"]}_covariance-baseline_downsample_f4_f45_lcmv.pkl').as_posix()

    def get_lcmv_f4_f45_spatial_temporal_contrast_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}'
            f'_{kwargs["target_event_id"]}_covariance-baseline_downsample_f4_f45_lcmv_contrast.pkl').as_posix()

    def get_lcmv_spatio_temporal_cluster_1samp_test_fname(self, **kwargs):
        return f'lcmv_{kwargs["freq"]}_{kwargs["target_event"]}_cov-baseline_downsample_spatio_temporal_cluster_1samp_test.pkl'

    def get_plot_lcmv_freq_f4_f45_clusters_fname(self, **kwargs):
        figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
        return figures_d.joinpath(
            f'lcmv_{kwargs["freq"]}_{kwargs["target_event_id"]}_'
            'f4_f45_covariance_baseline_downsample_clusters.png').as_posix()

    def get_lcmv_solution(self, subject, task_selected, target_event_id):
        # Make sure the number of noise epochs is the same as data epochs
        # note also need to apply downsample to noise epochs
        tmin, tmax, tmin_plot, tmax_plot = self.get_lcmv_event_epochs_config()
        # epochs, raw, events = \
            # self.get_lcmv_event_epochs_encoder(subject,
                                               # task_selected,
                                               # target_event_id)
        epochs, raw, events = \
            self.get_lcmv_event_epochs_encoder_resampled(
                subject, task_selected, target_event_id, new_sampling_rate=500)
        fwd_fname = self.get_fwd_fname(subject=subject)
        forward = mne.read_forward_solution(fwd_fname)
        # Setting frequency bins as in Dalal et al. 2008 (high gamma was subdivided)
        # freq_bins = [(4, 12), (12, 30), (30, 55), (65, 299)]  # Hz
        # corresponding to get_lcmv_fname
        iter_freqs = [
            ('Theta', 4, 7),
            ('Alpha', 8, 12),
            ('Beta', 13, 25),
            ('Gamma', 30, 45)
        ]
        freq_bins = [(i[1], i[2]) for i in iter_freqs]
        win_lengths = [0.3, 0.2, 0.15, 0.1]  # s
        # Setting the time step
        tstep = 0.05
        # Setting the whitened data covariance regularization parameter
        data_reg = 0.001
        # Subtract evoked response prior to computation?
        subtract_evoked = False
        # Calculating covariance from empty room noise. To use baseline data as noise
        # substitute raw for raw_noise, epochs.events for epochs_noise.events, tmin for
        # desired baseline length, and 0 for tmax_plot.
        # Note, if using baseline data, the averaged evoked response in the baseline
        # period should be flat.
        noise_covs = []
        event_id = 254  # use calibrated with first stimuli
        raw.load_data()
        for (l_freq, h_freq) in freq_bins:
            raw_band = raw.copy()
            #raw_band.load_data()
            raw_band.filter(l_freq, h_freq, n_jobs=1, fir_design='firwin')
            epochs_band = mne.Epochs(raw_band, events, event_id,
                                     tmin=tmin_plot, tmax=0, baseline=None,
                                     proj=True, picks='meg')
            noise_cov = mne.compute_covariance(epochs_band, method='auto', rank=None)
            noise_covs.append(noise_cov)
            del raw_band  # to save memory
        # Computing LCMV solutions for time-frequency windows in a label in source
        # space for faster computation, use label=None for full solution
        stcs = mne.beamformer.tf_lcmv(
            epochs, forward, noise_covs, tmin, tmax, tstep, win_lengths,
            freq_bins=freq_bins, subtract_evoked=subtract_evoked, n_jobs=4,
            reg=data_reg, rank=None)
        return stcs

    def lcmv_spatio_temporal_cluster_1samp_test(self, subjects, target_event):
        """Target_event: sti2"""
        from devp_basicio import save_list_pkl
        from scipy import stats as stats
        from mne.stats import spatio_temporal_cluster_1samp_test, ttest_1samp_no_p
        #from mne.stats import permutation_cluster_1samp_test
        from functools import partial
        _list, n_subjects = self.load_stacking_lcmv_spatial_temporal_contrast(subjects, target_event)
        self.logger.info('Computing connectivity.')
        src_fname = '/home/foucault/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-ico-5-src.fif'
        src = mne.read_source_spaces(src_fname)
        connectivity = mne.spatial_src_connectivity(src)
        # p_threshold = 1.0e-1
        # t_threshold = -stats.distributions.t.ppf(
            # p_threshold / 2., n_subjects - 1)
        freq_iter = _list[0].keys()
        freq_dict = self._transform_to_training(_list)
        sigma = 1e-3
        n_permutations = 'all'
        stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
        threshold_tfce = dict(start=0, step=0.2)
        for freq in freq_iter:
            # XXX debug
            # if freq == 'Theta':
                # continue
            # if freq == 'Alpha':
                # continue
            X = freq_dict[freq]
            X = np.transpose(X, [0, 2, 1])
            self.logger.info(f'Clustering in band: {freq}')
            # t_tfce_hat, clusters, p_tfce_hat, H0 = clu = \
                # spatio_temporal_cluster_1samp_test(
                    # X, n_jobs=1, threshold=threshold_tfce, connectivity=None,
                    # n_permutations=n_permutations, stat_fun=stat_fun_hat, buffer_size=None)
            # save_list_pkl(clu, 'sti2_tfce_hat_permutation_cluster_1samp_test.pkl', logger=self.logger)
            #T_obs, clusters, cluster_p_values, H0 = clu = \
            t_tfce_hat, clusters, p_tfce_hat, H0 = clu = \
                spatio_temporal_cluster_1samp_test(
                    X, connectivity=connectivity, n_jobs=6,
                    threshold=threshold_tfce, stat_fun=stat_fun_hat,
                    buffer_size=None, verbose=True)
            lcmv_spatio_temporal_cluster_1samp_test_fname = \
                self.get_lcmv_spatio_temporal_cluster_1samp_test_fname(
                    target_event=target_event, freq=freq)
            save_list_pkl(clu, lcmv_spatio_temporal_cluster_1samp_test_fname, logger=self.logger)
            # Now select the clusters that are sig. at p < 0.05 (note that this value
            # is multiple-comparisons corrected).
            # good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

    def plot_lcmv_spatio_temporal_cluster_1samp_test(self, target_event):
        from mayavi import mlab
        mlab.options.offscreen = True
        from devp_basicio import load_list_pkl
        from mne.stats import summarize_clusters_stc
        target_event_id = target_event
        iter_freqs = [
            ('Theta', 4, 7),
            ('Alpha', 8, 12),
            ('Beta', 13, 25),
            ('Gamma', 30, 45)
        ]
        freq_bins = [i[0] for i in iter_freqs]
        for freq in freq_bins:
            lcmv_spatio_temporal_cluster_1samp_test_fname = \
                self.get_lcmv_spatio_temporal_cluster_1samp_test_fname(
                    target_event=target_event, freq=freq)
            clu = load_list_pkl(lcmv_spatio_temporal_cluster_1samp_test_fname, logger=self.logger)
            T_obs, clusters, cluster_p_values, H0 = clu
            self.logger.info('Visualizing clusters.')
            # Now let's build a convenient representation of each cluster, where each
            # cluster becomes a "time point" in the SourceEstimate
            #epochs = self.get_epochs_encoder_sti2()
            tstep = 0.002 # 1/epochs.info['sfreq']
            src_fname = '/home/foucault/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-ico-5-src.fif'
            src = mne.read_source_spaces(src_fname)
            fsave_vertices = [s['vertno'] for s in src]
            # stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep, p_thresh=0.05,
                                                        # vertices=fsave_vertices,
                                                        # subject='fsaverage')

            # Let's actually plot the first "time point" in the SourceEstimate, which
            # shows all the clusters, weighted by duration
            # blue blobs are for condition A < condition B, red for A > B
            # brain = stc_all_cluster_vis.plot(
                # hemi='both', views='lateral', subjects_dir=self.SUBJECTS_DIR,
                # time_label='Duration significant (ms)', size=(800, 800),
                # smoothing_steps=5, clim=dict(kind='value', pos_lims=[0, 1, 40]))
            p_thresh = cluster_p_values.min() + 0.02
            self.logger.info(f'min cluster_p_values: {cluster_p_values.min()}')
            self.logger.info(f'p_thresh: {p_thresh}')
            stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep, p_thresh=p_thresh, vertices=fsave_vertices, subject='fsaverage')
            brain = stc_all_cluster_vis.plot(hemi='split', views='med', subjects_dir=self.SUBJECTS_DIR, time_label='Duration significant (ms)', size=(800, 800), smoothing_steps=5)
            w_png = self.get_plot_lcmv_freq_f4_f45_clusters_fname(
                freq=freq, target_event_id=target_event_id)
            self.logger.info(f'saveing fig: {w_png}')
            brain.save_image(w_png)
            # __import__('IPython').embed()
            # __import__('sys').exit()


class SourceSpaceStat8(SourceSpaceStat7):
    """5D time-frequency beamforming based on LCMV
    with covariance from sti1 baseline
    Downsample for TFCE, without hat
    """
    def __init__(self):
        super().__init__()
        self.logger = logger

    def get_lcmv_spatio_temporal_cluster_1samp_test_fname(self, **kwargs):
        return f'lcmv_{kwargs["freq"]}_{kwargs["target_event"]}_cov-baseline_downsample_wo_hat_spatio_temporal_cluster_1samp_test.pkl'

    def get_plot_lcmv_freq_f4_f45_clusters_fname(self, **kwargs):
        figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
        return figures_d.joinpath(
            f'lcmv_{kwargs["freq"]}_{kwargs["target_event_id"]}_'
            'f4_f45_covariance_baseline_downsample_wo_hat_clusters.png').as_posix()

    def lcmv_spatio_temporal_cluster_1samp_test(self, subjects, target_event):
        """Target_event: sti2"""
        from devp_basicio import save_list_pkl
        from scipy import stats as stats
        from mne.stats import spatio_temporal_cluster_1samp_test, ttest_1samp_no_p
        #from mne.stats import permutation_cluster_1samp_test
        from functools import partial
        _list, n_subjects = self.load_stacking_lcmv_spatial_temporal_contrast(subjects, target_event)
        self.logger.info('Computing connectivity.')
        src_fname = '/home/foucault/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-ico-5-src.fif'
        src = mne.read_source_spaces(src_fname)
        connectivity = mne.spatial_src_connectivity(src)
        # p_threshold = 1.0e-1
        # t_threshold = -stats.distributions.t.ppf(
            # p_threshold / 2., n_subjects - 1)
        freq_iter = _list[0].keys()
        freq_dict = self._transform_to_training(_list)
        sigma = 1e-3
        n_permutations = 'all'
        stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
        threshold_tfce = dict(start=0, step=0.2)
        for freq in freq_iter:
            # XXX debug
            # if freq == 'Theta':
                # continue
            # if freq == 'Alpha':
                # continue
            X = freq_dict[freq]
            X = np.transpose(X, [0, 2, 1])
            self.logger.info(f'Clustering in band: {freq}')
            # t_tfce_hat, clusters, p_tfce_hat, H0 = clu = \
                # spatio_temporal_cluster_1samp_test(
                    # X, n_jobs=1, threshold=threshold_tfce, connectivity=None,
                    # n_permutations=n_permutations, stat_fun=stat_fun_hat, buffer_size=None)
            # save_list_pkl(clu, 'sti2_tfce_hat_permutation_cluster_1samp_test.pkl', logger=self.logger)
            #T_obs, clusters, cluster_p_values, H0 = clu = \
            t_tfce, clusters, p_tfce, H0 = clu = \
                spatio_temporal_cluster_1samp_test(
                    X, connectivity=connectivity, n_jobs=6,
                    threshold=threshold_tfce,
                    buffer_size=None, verbose=True)
            lcmv_spatio_temporal_cluster_1samp_test_fname = \
                self.get_lcmv_spatio_temporal_cluster_1samp_test_fname(
                    target_event=target_event, freq=freq)
            save_list_pkl(clu, lcmv_spatio_temporal_cluster_1samp_test_fname, logger=self.logger)
            # Now select the clusters that are sig. at p < 0.05 (note that this value
            # is multiple-comparisons corrected).
            # good_cluster_inds = np.where(cluster_p_values < 0.05)[0]


class SourceSpaceStat9(SourceSpaceStat8):
    """5D time-frequency beamforming based on LCMV
    with covariance from sti1 baseline
    Downsample for TFCE, without hat
    Time lock with maintainance; id: 239
    """
    def __init__(self):
        super().__init__()
        self.logger = logger

    def get_lcmv_event_epochs_config(self):
        tmin, tmax = -0.5, 4
        tmin_plot, tmax_plot = -0.1, 3.75
        return tmin, tmax, tmin_plot, tmax_plot


class SourceSpaceStat9(SourceSpaceStat8):
    """5D time-frequency beamforming based on DICS
    with covariance from sti1 baseline
    Downsample for TFCE, without hat
    Time lock with maintainance; id: 251
    """
    def __init__(self):
        super().__init__()
        self.logger = logger

    def get_dics_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}_{kwargs["task_selected"]}'
            f'_{kwargs["target_event"]}_dics.pkl').as_posix()

    def get_dics_spatial_temporal_contrast_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}'
            f'_{kwargs["target_event"]}_dics_contrast.pkl').as_posix()

    def get_dics_spatio_temporal_cluster_1samp_test_fname(self, **kwargs):
        return f'dics_{kwargs["target_event"]}_spatio_temporal_cluster_1samp_test.pkl'

    def get_plot_dics_freq_clusters_fname(self, **kwargs):
        figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
        return figures_d.joinpath(
            f'dics_{kwargs["freq"]}_{kwargs["target_event"]}_'
            'downsample_clusters.png').as_posix()

    def get_plot_dics_freq_clusters_single_fname(self, **kwargs):
        figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
        return figures_d.joinpath(
            f'dics_{kwargs["freq"]}_{kwargs["target_event"]}_'
            'downsample_single_clusters.png').as_posix()

    def get_dics_event_epochs_config(self):
        tmin, tmax = -0.5, 2
        tmin_plot, tmax_plot = -0.1, 1.75
        return tmin, tmax, tmin_plot, tmax_plot

    def get_dics_event_epochs_encoder(self, subject, task_selected, target_event):
        """Get epochs time lock to encoder stimulus 2"""
        from devp_source_space import reconstruct_raw
        self.logger.info(f'Get epochs time lock to encoder stimulus id {target_event}')
        recon_raw, events, filt_raw_em = reconstruct_raw(
            subject, task_selected)
        event_ids = list(np.unique(events[:,2]))
        # fine-tune time length
        tmin, tmax, tmin_plot, tmax_plot = self.get_dics_event_epochs_config()
        baseline = None
        logger.info(f"target event_id: {target_event}")
        # remove the reject=dict(mag=1.5e-12) option, too much data point lost
        epochs = mne.Epochs(recon_raw, events, target_event, tmin, tmax,
                            baseline=baseline, picks='meg',
                            preload=False)
        return epochs

    def get_dics_event_epochs_encoder_resampled(self, **kwargs):
        """Get epochs time lock to encoder stimulus target_event
        Resampling"""
        from devp_source_space import reconstruct_raw_w_file
        subject=kwargs['subject']
        task_selected=kwargs['task_selected']
        target_event=kwargs['target_event']
        new_sampling_rate=kwargs['new_sampling_rate']
        freq_bins=kwargs['freq_bins']
        self.logger.info(f'Get epochs time lock to encoder stimulus {target_event}...')
        # recon_raw, events, filt_raw_em = reconstruct_raw(subject, task_selected)
        # new version
        recon_raw, events, filt_raw_em = reconstruct_raw_w_file(
            subject, task_selected, base_file=self.mne_derivatives_dname,
            logger=self.logger)
        recon_raw.load_data().filter(freq_bins[0], freq_bins[1])
        self.logger.info(f"Original sampling rate:{recon_raw.info['sfreq']} Hz")
        raw_resampled, events = recon_raw.copy().resample(sfreq=500, events=events)
        self.logger.info(f"New sampling rate:{raw_resampled.info['sfreq']} Hz")
        event_ids = list(np.unique(events[:,2]))
        # fine-tune time length
        tmin, tmax, tmin_plot, tmax_plot = self.get_dics_event_epochs_config()
        baseline = None
        logger.info(f"target event: {target_event}")
        # remove the reject=dict(mag=1.5e-12) option, too much data point lost
        #epochs = mne.Epochs(recon_raw, events, target_event, tmin, tmax,
        epochs = mne.Epochs(raw_resampled, events, target_event, tmin, tmax,
                            baseline=baseline, picks='meg',
                            preload=False)
        return epochs, raw_resampled, events


    # def get_dics_noise_cov(self, subject, target_event_id):
        # """Prepare empty room noise covariance used in lcmv method"""
        # from mne.event import make_fixed_length_events
        # self.logger.info('Prepare empty room noise covariance used in lcmv method')
        # raw_empty_room = filter_raw_empty_room(
            # subject, logger=self.logger)
        # # Create artificial events for empty room noise data
        # events_noise = make_fixed_length_events(raw_empty_room, target_event_id, duration=1.)
        # tmin, tmax, tmin_plot, tmax_plot = self.get_lcmv_event_epochs_config()
        # epochs_noise = mne.Epochs(raw_empty_room, events_noise, target_event_id, tmin, tmax)
        # return epochs_noise, raw_empty_room

    def _gen_dics(self, active_win, baseline_win, epochs, fwd):
        freqs = np.logspace(np.log10(12), np.log10(30), 9)
        tmin, tmax, tmin_plot, tmax_plot = self.get_dics_event_epochs_config()
        from mne.time_frequency import csd_morlet
        from mne.beamformer import make_dics, apply_dics_csd
        epochs.load_data()
        csd = csd_morlet(
            epochs, freqs, tmin=-0.5, tmax=active_win[1],
            decim=1, n_jobs=6)
        csd_baseline = csd_morlet(epochs, freqs, tmin=baseline_win[0],
                                  tmax=baseline_win[1], decim=1, n_jobs=6)
        csd_ers = csd_morlet(epochs, freqs, tmin=active_win[0],
                             tmax=active_win[1], decim=1, n_jobs=6)
        filters = make_dics(epochs.info, fwd, csd.mean(), pick_ori='max-power')
        stc_base, freqs = apply_dics_csd(csd_baseline.mean(), filters)
        stc_act, freqs = apply_dics_csd(csd_ers.mean(), filters)
        stc_act /= stc_base
        return stc_act

    def get_dics_solution(self, subject, task_selected, target_event):
        # Make sure the number of noise epochs is the same as data epochs
        tmin, tmax, tmin_plot, tmax_plot = self.get_dics_event_epochs_config()
        fwd_fname = self.get_fwd_fname(subject=subject)
        fwd = mne.read_forward_solution(fwd_fname)
        active_win = (0.5, 2)
        baseline_win = (-0.5, 0)
        # Setting frequency bins as in Dalal et al. 2008 (high gamma was subdivided)
        # freq_bins = [(4, 12), (12, 30), (30, 55), (65, 299)]  # Hz
        # corresponding to get_lcmv_fname
        iter_freqs = [
            ('Theta', 4, 7),
            ('Alpha', 8, 12),
            ('Beta', 13, 25),
            ('Gamma', 30, 45)
        ]
        freq_bins = [(i[1], i[2]) for i in iter_freqs]
        # # corresponding to get_lcmv_f4_f45_fname
        # win_lengths = [0.3, 0.2, 0.15, 0.1]  # s
        # # Setting the time step
        # tstep = 0.05
        # # Setting the whitened data covariance regularization parameter
        # data_reg = 0.001
        # # Subtract evoked response prior to computation?
        # subtract_evoked = False
        # # Calculating covariance from empty room noise. To use baseline data as noise
        # # substitute raw for raw_noise, epochs.events for epochs_noise.events, tmin for
        # # desired baseline length, and 0 for tmax_plot.
        # # Note, if using baseline data, the averaged evoked response in the baseline
        # # period should be flat.
        # noise_covs = []
        stcs = list()
        for (l_freq, h_freq) in freq_bins:
            event_epochs_kwargs = dict(
                subject=subject, task_selected=task_selected,
                target_event=target_event, new_sampling_rate=500,
                freq_bins=(l_freq, h_freq))
            epochs, raw_resampled, events = \
                self.get_dics_event_epochs_encoder_resampled(
                    **event_epochs_kwargs)
            stc = self._gen_dics(active_win, baseline_win, epochs, fwd)
            stcs.append(stc)
        return stcs
            # raw_band = raw_noise.copy()
            # raw_band.filter(l_freq, h_freq, n_jobs=1, fir_design='firwin')
            # epochs_band = mne.Epochs(raw_band, epochs_noise.events, target_event_id,
                                     # tmin=tmin_plot, tmax=tmax_plot, baseline=None,
                                     # proj=True, picks='meg')
            # noise_cov = mne.compute_covariance(epochs_band, method='auto', rank=None)
            # noise_covs.append(noise_cov)
            # del raw_band  # to save memory
        # # Computing LCMV solutions for time-frequency windows in a label in source
        # # space for faster computation, use label=None for full solution
        # stcs = mne.beamformer.tf_lcmv(
            # epochs, forward, noise_covs, tmin, tmax, tstep, win_lengths,
            # freq_bins=freq_bins, subtract_evoked=subtract_evoked,
            # reg=data_reg, rank=None)
        # return stcs

    def save_dics_solution(self, subject, task_selected, target_event):
        from devp_basicio import save_list_pkl
        stcs = self.get_dics_solution(subject, task_selected, target_event)
        dics_fname = self.get_dics_fname(
            subject=subject, task_selected=task_selected,
            target_event=target_event)
        save_list_pkl(stcs, dics_fname, logger=self.logger)

    def get_dics_spatial_temporal_contrast(self, subject, target_event):
        from devp_basicio import load_list_pkl
        task_stcs_dict = dict()
        freq_stcs_contrast_dict = dict()
        iter_freqs = [
            ('Theta', 4, 7),
            ('Alpha', 8, 12),
            ('Beta', 13, 25),
            ('Gamma', 30, 45)
        ]
        freq_bins = [i[0] for i in iter_freqs]
        for task_selected in self.task_list:
            dics_fname = self.get_dics_fname(
                subject=subject, task_selected=task_selected,
                target_event=target_event)
            stcs = load_list_pkl(dics_fname, logger=self.logger)
            task_stcs_dict[task_selected] = stcs
        inv_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}-inv.fif')
        self.logger.info(f'Loading inv_fname: {inv_fname}')
        inv = mne.minimum_norm.read_inverse_operator(inv_fname)
        morph_mat, n_vertices_fsave = self.get_morphed_mat(subject, inv)
        for i in range(len(stcs)):
            freq_stcs_contrast_dict[freq_bins[i]] = \
                morph_mat.dot(task_stcs_dict['spatial'][i].data) -\
                morph_mat.dot(task_stcs_dict['temporal'][i].data)
        return freq_stcs_contrast_dict

    def save_dics_spatial_temporal_contrast(self, subject, target_event):
        from devp_basicio import save_list_pkl
        freq_stcs_contrast_dict = \
            self.get_dics_spatial_temporal_contrast(subject, target_event)
        contrast_fname = self.get_dics_spatial_temporal_contrast_fname(
            subject=subject,
            target_event=target_event)
        save_list_pkl(freq_stcs_contrast_dict,
                      contrast_fname, logger=self.logger)

    def load_dics_spatial_temporal_contrast(self, subject, target_event):
        from devp_basicio import load_list_pkl
        contrast_fname = \
        self.get_dics_spatial_temporal_contrast_fname(
            subject=subject,
            target_event=target_event)
        freq_stcs_contrast_dict = load_list_pkl(
            contrast_fname, logger=self.logger)
        return freq_stcs_contrast_dict

    def load_stacking_dics_spatial_temporal_contrast(self, subjects, target_event):
        """Target_event_id"""
        from devp_basicio import load_list_pkl
        _list = list()
        for subject in subjects:
            self.logger.info(f'Processing {subject} ...')
            freq_stcs_contrast_dict = \
            self.load_dics_spatial_temporal_contrast(
                subject, target_event)
            _list.append(freq_stcs_contrast_dict)
        return _list, len(_list)

    def _transform_to_training(self, _list):
        """Transform list of dicts of frquency to training"""
        from itertools import cycle
        freq_iter = _list[0].keys()
        array_dict = dict()
        for freq in freq_iter:
            array_dict[freq] = np.stack([j[freq] for j, freq in zip(_list, cycle([freq]))])
        return array_dict

    def dics_spatio_temporal_cluster_1samp_test(self, subjects, target_event):
        """Target_event: sti3"""
        from devp_basicio import save_list_pkl
        from scipy import stats as stats
        from mne.stats import spatio_temporal_cluster_1samp_test
        from functools import partial
        _list, n_subjects = self.load_stacking_dics_spatial_temporal_contrast(subjects, target_event)
        self.logger.info('Computing connectivity.')
        src_fname = '/home/foucault/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-ico-5-src.fif'
        src = mne.read_source_spaces(src_fname)
        connectivity = mne.spatial_src_connectivity(src)
        # p_threshold = 1.0e-1
        # t_threshold = -stats.distributions.t.ppf(
            # p_threshold / 2., n_subjects - 1)
        freq_iter = _list[0].keys()
        freq_dict = self._transform_to_training(_list)
        # freq_dict['Theta'].shape: (16, 20484, 1)
        # to see the structure
        sigma = 1e-3
        n_permutations = 'all'
        # stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
        threshold_tfce = dict(start=0, step=0.2)
        for freq in freq_iter:
            # XXX debug
            # if freq == 'Theta':
                # continue
            # if freq == 'Alpha':
                # continue
            X = freq_dict[freq]
            X = np.transpose(X, [0, 2, 1])
            self.logger.info(f'Clustering in band: {freq}')
            # t_tfce_hat, clusters, p_tfce_hat, H0 = clu = \
                # spatio_temporal_cluster_1samp_test(
                    # X, n_jobs=1, threshold=threshold_tfce, connectivity=None,
                    # n_permutations=n_permutations, stat_fun=stat_fun_hat, buffer_size=None)
            # save_list_pkl(clu, 'sti2_tfce_hat_permutation_cluster_1samp_test.pkl', logger=self.logger)
            #T_obs, clusters, cluster_p_values, H0 = clu = \
            t_tfce, clusters, p_tfce, H0 = clu = \
                spatio_temporal_cluster_1samp_test(
                    X, connectivity=connectivity, n_jobs=6,
                    threshold=threshold_tfce,
                    buffer_size=None, verbose=True)
            lcmv_spatio_temporal_cluster_1samp_test_fname = \
                self.get_dics_spatio_temporal_cluster_1samp_test_fname(
                    target_event=target_event, freq=freq)
            save_list_pkl(clu, lcmv_spatio_temporal_cluster_1samp_test_fname, logger=self.logger)
            # Now select the clusters that are sig. at p < 0.05 (note that this value
            # is multiple-comparisons corrected).
            # good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

    def plot_dics_spatio_temporal_cluster_1samp_test(self, target_event):
        from mayavi import mlab
        mlab.options.offscreen = True
        from devp_basicio import load_list_pkl
        from mne.stats import summarize_clusters_stc
        target_event = target_event
        iter_freqs = [
            ('Theta', 4, 7),
            ('Alpha', 8, 12),
            ('Beta', 13, 25),
            ('Gamma', 30, 45)
        ]
        freq_bins = [i[0] for i in iter_freqs]
        for freq in freq_bins:
            dics_spatio_temporal_cluster_1samp_test_fname = \
                self.get_dics_spatio_temporal_cluster_1samp_test_fname(
                    target_event=target_event, freq=freq)
            clu = load_list_pkl(dics_spatio_temporal_cluster_1samp_test_fname, logger=self.logger)
            T_obs, clusters, cluster_p_values, H0 = clu
            self.logger.info('Visualizing clusters.')
            # Now let's build a convenient representation of each cluster, where each
            # cluster becomes a "time point" in the SourceEstimate
            #epochs = self.get_epochs_encoder_sti2()
            tstep = 0.001 # 1/epochs.info['sfreq']
            src_fname = '/home/foucault/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-ico-5-src.fif'
            src = mne.read_source_spaces(src_fname)
            fsave_vertices = [s['vertno'] for s in src]
            # stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep, p_thresh=0.05,
                                                        # vertices=fsave_vertices,
                                                        # subject='fsaverage')

            # Let's actually plot the first "time point" in the SourceEstimate, which
            # shows all the clusters, weighted by duration
            # blue blobs are for condition A < condition B, red for A > B
            # brain = stc_all_cluster_vis.plot(
                # hemi='both', views='lateral', subjects_dir=self.SUBJECTS_DIR,
                # time_label='Duration significant (ms)', size=(800, 800),
                # smoothing_steps=5, clim=dict(kind='value', pos_lims=[0, 1, 40]))
            p_thresh = cluster_p_values.min() + 0.02
            self.logger.info(f'min cluster_p_values: {cluster_p_values.min()}')
            self.logger.info(f'p_thresh: {p_thresh}')
            stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep, p_thresh=p_thresh, vertices=fsave_vertices, subject='fsaverage')
            brain = stc_all_cluster_vis.plot(hemi='split', views='med', subjects_dir=self.SUBJECTS_DIR, time_label='Duration significant (ms)', size=(800, 800), smoothing_steps=5)
            w_png = self.get_plot_dics_freq_clusters_fname(
                freq=freq, target_event=target_event)
            self.logger.info(f'saveing fig: {w_png}')
            brain.save_image(w_png)
            # __import__('IPython').embed()
            # __import__('sys').exit()

    def plot_dics_spatio_temporal_cluster_1samp_test_single(self, subject, target_event):
        """Visualize single subject data"""
        from mayavi import mlab
        mlab.options.offscreen = True
        from devp_basicio import load_list_pkl
        from mne.stats import summarize_clusters_stc
        target_event = target_event
        dics_fname = self.get_dics_fname(
            subject=subject, task_selected=self.task_list[0],
            target_event=target_event)
        stcs = load_list_pkl(dics_fname, logger=self.logger)
        iter_freqs = [
            ('Theta', 4, 7),
            ('Alpha', 8, 12),
            ('Beta', 13, 25),
            ('Gamma', 30, 45)
        ]
        freq_bins = [i[0] for i in iter_freqs]
        for i, freq in enumerate(freq_bins):
            brain = stcs[i].plot(hemi='split', views='lat', subjects_dir=self.SUBJECTS_DIR,
                    subject=subject)
            w_png = self.get_plot_dics_freq_clusters_single_fname(
                freq=freq, target_event=target_event)
            self.logger.info(f'saveing fig: {w_png}')
            brain.save_image(w_png)
            # dics_spatio_temporal_cluster_1samp_test_fname = \
                # self.get_dics_spatio_temporal_cluster_1samp_test_fname(
                    # target_event=target_event, freq=freq)
            # clu = load_list_pkl(dics_spatio_temporal_cluster_1samp_test_fname, logger=self.logger)
            # T_obs, clusters, cluster_p_values, H0 = clu
            # self.logger.info('Visualizing clusters.')
            # # Now let's build a convenient representation of each cluster, where each
            # # cluster becomes a "time point" in the SourceEstimate
            # #epochs = self.get_epochs_encoder_sti2()
            # tstep = 0.001 # 1/epochs.info['sfreq']
            # src_fname = '/home/foucault/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-ico-5-src.fif'
            # src = mne.read_source_spaces(src_fname)
            # fsave_vertices = [s['vertno'] for s in src]
            # # stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep, p_thresh=0.05,
                                                        # # vertices=fsave_vertices,
                                                        # # subject='fsaverage')

            # # Let's actually plot the first "time point" in the SourceEstimate, which
            # # shows all the clusters, weighted by duration
            # # blue blobs are for condition A < condition B, red for A > B
            # # brain = stc_all_cluster_vis.plot(
                # # hemi='both', views='lateral', subjects_dir=self.SUBJECTS_DIR,
                # # time_label='Duration significant (ms)', size=(800, 800),
                # # smoothing_steps=5, clim=dict(kind='value', pos_lims=[0, 1, 40]))
            # p_thresh = cluster_p_values.min() + 0.02
            # self.logger.info(f'min cluster_p_values: {cluster_p_values.min()}')
            # self.logger.info(f'p_thresh: {p_thresh}')
            # stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep, p_thresh=p_thresh, vertices=fsave_vertices, subject='fsaverage')
            # brain = stc_all_cluster_vis.plot(hemi='split', views='med', subjects_dir=self.SUBJECTS_DIR, time_label='Duration significant (ms)', size=(800, 800), smoothing_steps=5)
            # w_png = self.get_plot_dics_freq_clusters_fname(
                # freq=freq, target_event=target_event)
            # self.logger.info(f'saveing fig: {w_png}')
            # brain.save_image(w_png)
            # __import__('IPython').embed()
            # __import__('sys').exit()



def main_source_space_exp_5():
    """Lcmv with covariance from sti1 baseline"""
    from devp_basicio import define_id_by_aging_dict
    from devp_source_space import get_subjects_list
    sss = SourceSpaceStat5()
    sss.logger = logger
    subjects = get_subjects_list()
    # for subject in subjects:
        # logger.info(f'Processing {subject} ...')
        # for task_selected in sss.task_list:
            # sss.save_lcmv_solution(subject, task_selected, 253)
        # sss.save_lcmv_spatial_temporal_contrast(subject, 253)

    # aging_dict = define_id_by_aging_dict()
    # _subjects = list()
    # for subject in subjects:
            # # pick just young
        # if int(subject[-3:]) not in aging_dict['young']:
            # logger.info(f'Filtering out {subject} ...')
            # continue
        # _subjects.append(subject)
        # logger.info(f'Appending {subject} ...')
    # subjects = _subjects
    # sss.lcmv_spatio_temporal_cluster_1samp_test(subjects, 253)
    sss.plot_lcmv_spatio_temporal_cluster_1samp_test(253)
    __import__('IPython').embed()
    __import__('sys').exit()


def main_source_space_exp_6():
    """Lcmv with covariance from sti1 baseline
    Set target event to third encoding stimuli (sti3)
    """
    from devp_basicio import define_id_by_aging_dict
    from devp_source_space import get_subjects_list
    sss = SourceSpaceStat5()
    sss.logger = logger
    subjects = get_subjects_list()
    for subject in subjects:
        logger.info(f'Processing {subject} ...')
        for task_selected in sss.task_list:
            sss.save_lcmv_solution(subject, task_selected, 251)
        sss.save_lcmv_spatial_temporal_contrast(subject, 251)
    # select group
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
    sss.lcmv_spatio_temporal_cluster_1samp_test(subjects, 251)
    sss.plot_lcmv_spatio_temporal_cluster_1samp_test(251)


def main_source_space_exp_7():
    """Lcmv with covariance from sti1 baseline
    Set target event to third encoding stimuli (sti4)
    """
    from devp_basicio import define_id_by_aging_dict
    from devp_source_space import get_subjects_list
    sss = SourceSpaceStat5()
    sss.logger = logger
    subjects = get_subjects_list()
    for subject in subjects:
        logger.info(f'Processing {subject} ...')
        for task_selected in sss.task_list:
            sss.save_lcmv_solution(subject, task_selected, 247)
        sss.save_lcmv_spatial_temporal_contrast(subject, 247)
    # select group
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
    sss.lcmv_spatio_temporal_cluster_1samp_test(subjects, 247)
    sss.plot_lcmv_spatio_temporal_cluster_1samp_test(247)


def main_source_space_exp_8():
    """Lcmv with covariance from sti1 baseline
    substract from evoked when beamforming
    """
    from devp_basicio import define_id_by_aging_dict
    from devp_source_space import get_subjects_list
    sss = SourceSpaceStat6()
    sss.logger = logger
    subjects = get_subjects_list()
    for subject in subjects:
        logger.info(f'Processing {subject} ...')
        for task_selected in sss.task_list:
            sss.save_lcmv_solution(subject, task_selected, 253)
        sss.save_lcmv_spatial_temporal_contrast(subject, 253)
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
    sss.lcmv_spatio_temporal_cluster_1samp_test(subjects, 253)
    sss.plot_lcmv_spatio_temporal_cluster_1samp_test(253)


def main_source_space_exp_9():
    """Lcmv with covariance from sti1 baseline
    resample
    """
    from devp_basicio import define_id_by_aging_dict
    from devp_source_space import get_subjects_list
    sss = SourceSpaceStat7()
    sss.logger = logger
    subjects = get_subjects_list()
    # for subject in subjects:
        # logger.info(f'Processing {subject} ...')
        # for task_selected in sss.task_list:
            # sss.save_lcmv_solution(subject, task_selected, 253)
        # sss.save_lcmv_spatial_temporal_contrast(subject, 253)
    # aging_dict = define_id_by_aging_dict()
    # _subjects = list()
    # for subject in subjects:
            # # pick just young
        # if int(subject[-3:]) not in aging_dict['young']:
            # logger.info(f'Filtering out {subject} ...')
            # continue
        # _subjects.append(subject)
        # logger.info(f'Appending {subject} ...')
    # subjects = _subjects
    # sss.lcmv_spatio_temporal_cluster_1samp_test(subjects, 253)
    sss.plot_lcmv_spatio_temporal_cluster_1samp_test(253)


def main_source_space_exp_10():
    """Lcmv with covariance from sti1 baseline
    resample, without hat
    """
    from devp_basicio import define_id_by_aging_dict
    from devp_source_space import get_subjects_list
    sss = SourceSpaceStat8()
    sss.logger = logger
    subjects = get_subjects_list()
    # for subject in subjects:
        # logger.info(f'Processing {subject} ...')
        # for task_selected in sss.task_list:
            # sss.save_lcmv_solution(subject, task_selected, 253)
        # sss.save_lcmv_spatial_temporal_contrast(subject, 253)
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
    sss.lcmv_spatio_temporal_cluster_1samp_test(subjects, 253)
    sss.plot_lcmv_spatio_temporal_cluster_1samp_test(253)


def main_source_space_exp_11():
    """Lcmv with covariance from sti1 baseline
    resample, without hat
    time lock at sti3; evnet id: 251
    """
    from devp_basicio import define_id_by_aging_dict
    from devp_source_space import get_subjects_list
    sss = SourceSpaceStat8()
    sss.logger = logger
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
    # for subject in subjects:
        # logger.info(f'Processing {subject} ...')
        # for task_selected in sss.task_list:
            # sss.save_lcmv_solution(subject, task_selected, 251)
        # sss.save_lcmv_spatial_temporal_contrast(subject, 251)
    # sss.lcmv_spatio_temporal_cluster_1samp_test(subjects, 251)
    sss.plot_lcmv_spatio_temporal_cluster_1samp_test(251)


def main_source_space_exp_12():
    """Lcmv with covariance from sti1 baseline
    resample, without hat
    time lock at sti4; evnet id: 247
    """
    from devp_basicio import define_id_by_aging_dict
    from devp_source_space import get_subjects_list
    sss = SourceSpaceStat8()
    sss.logger = logger
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
    for subject in subjects:
        logger.info(f'Processing {subject} ...')
        for task_selected in sss.task_list:
            sss.save_lcmv_solution(subject, task_selected, 247)
        sss.save_lcmv_spatial_temporal_contrast(subject, 247)
    sss.lcmv_spatio_temporal_cluster_1samp_test(subjects, 247)
    #sss.plot_lcmv_spatio_temporal_cluster_1samp_test(247)


def main_source_space_exp_13():
    """Lcmv with covariance from sti1 baseline
    resample, without hat
    Time lock at maintainance; id: 239
    """
    from devp_basicio import define_id_by_aging_dict
    from devp_source_space import get_subjects_list
    sss = SourceSpaceStat9()
    sss.logger = logger
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
    for subject in subjects:
        logger.info(f'Processing {subject} ...')
        for task_selected in sss.task_list:
            sss.save_lcmv_solution(subject, task_selected, 239)
        sss.save_lcmv_spatial_temporal_contrast(subject, 239)
    sss.lcmv_spatio_temporal_cluster_1samp_test(subjects, 239)
    #sss.plot_lcmv_spatio_temporal_cluster_1samp_test(239)


def main_source_space_exp_14():
    """DICS with covariance from sti1 baseline
    resample, without hat
    time lock at sti3; evnet id: 251
    """
    from devp_basicio import define_id_by_aging_dict
    from devp_source_space import get_subjects_list
    sss = SourceSpaceStat9()
    sss.logger = logger
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
    # for subject in subjects:
        # logger.info(f'Processing {subject} ...')
        # for task_selected in sss.task_list:
            # sss.save_dics_solution(subject, task_selected, 251)
        # sss.save_dics_spatial_temporal_contrast(subject, 251)
    # sss.dics_spatio_temporal_cluster_1samp_test(subjects, 251)
    # sss.plot_dics_spatio_temporal_cluster_1samp_test(251)
    sss.plot_dics_spatio_temporal_cluster_1samp_test_single(subjects[0], 251)


def main_random_test():
    from devp_source_space import get_subjects_list
    sss = SourceSpaceStat5()
    sss.logger = logger
    subjects = get_subjects_list()
    epochs, recon_raw, events = \
        sss.get_lcmv_event_epochs_encoder(subjects[0], sss.task_list[0], 251)


def main():
    # to examine plot to see the values
    # main_source_space_exp_9()
    # main_source_space_exp_10()
    # main_source_space_exp_11()
    # main_source_space_exp_12()
    # main_source_space_exp_13()
    main_source_space_exp_14()


if __name__ == '__main__':
    main()
