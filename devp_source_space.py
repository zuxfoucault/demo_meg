import os
import loggy
import copy
from pathlib import Path
import pickle
import mne
import numpy as np
# import matplotlib.pyplot as plt
# import mne_bids
#from devp_basicio import _read_raw_bids
import subprocess
from devp_basicio import epochs_filter_by_trial

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


def devp_set_config():
    print(mne.get_config('MNE_USE_CUDA'))
    try:
        mne.set_config('MNE_USE_CUDA', True)
    except TypeError as err:
        print(err)


def devp_set_config2():
    print(mne.get_config('MNE_USE_CUDA'))
    try:
        mne.set_config('MNE_USE_CUDA', True)
    except TypeError as err:
        print(err)


def run_command(command, log_file=None):
    import pprint
    if log_file == None:
        proc = subprocess.Popen(command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        for line in proc.stdout:
            pprint.pprint(line)
    else:
        pass
        # with open(log_file, "wb") as f:
            # proc = subprocess.Popen(command,
                                    # stdout=subprocess.PIPE,
                                    # stderr=subprocess.STDOUT)
            # for line in proc.stdout:
                # f.write(line)
    if proc.wait() != 0:
        raise RuntimeError("command failed")


def check_dir_or_mkdir(_pathobj, logger=None):
    if not _pathobj.exists():
        if logger is None:
            print(f"Directory created {_pathobj.as_posix()}")
        else:
            logger.info(f"Directory created {_pathobj.as_posix()}")
        _pathobj.mkdir(parents=True, exist_ok=True)


def get_subjects_list(logger=None):
    if logger is None:
        logger = Logger()
    logger.info('Get a list of subjects')
    from devp_basicio import define_id_list
    id_list, excluded_id_list, concatenated_data_dict = define_id_list()
    excluded_id_list.append('229')
    excluded_id_list.append('212')
    excluded_id_list.append('216')
    # reason for exclusion:
    # 229: RuntimeError: Could not find neighbor for vertex 9940 / 10242
    # 212: RuntimeError: Surface inner skull is not completely inside surface outer skull
    # 216: RuntimeError: Surface inner skull is not completely inside surface outer skull
    subjects = [ # watch out type(_id)
        f'sub-{_id}' for _id in id_list if str(_id) not in excluded_id_list]
    logger.info(f'Excluding: {excluded_id_list}')
    logger.info(f'{len(excluded_id_list)} participants excluded')
    logger.info(f'Get {len(subjects)} participants from original {len(id_list)}')
    return subjects


def get_event_id_dict(logger=None):
    if logger is None:
        logger = Logger()
    logger.info('Get event id dictionary')
    return dict(
        {'sti1': 254, 'sti2': 253, 'sti3': 251, 'sti4': 247,
         'maintenance': 239, 'retrieval': 223})


def _read_raw_bids(subject=None, task_selected=None):
    from mne_bids import read_raw_bids, make_bids_basename
    from devp_basicio import (read_id_task_dict, define_id_list,
                              concatenate_raw_sub126, concatenate_raw_sub228)
    if subject == None:
        subject = 'sub-104'
    if task_selected == None:
        task_selected = 'spatial'
        #task_selected = 'temporal'
    subject_id = subject[-3:]
    id_list, exclude_list, data_concated_dict = define_id_list()
    if subject_id in data_concated_dict.keys():
        raw, events, event_id, raw_em =\
            data_concated_dict[subject_id](task_selected)
        return raw, events, event_id, raw_em
    df = read_id_task_dict()
    # type(df['sub_id'][0]): numpy.int64
    root_dir = "/home/foucault/data/rawdata/working_memory"
    column = ['sub_id', '0', '1', '2']
    task_desc_dict = dict(zip(['subject', 'spatial', 'temporal', 'mis'], column))
    task = df.loc[df['sub_id']==int(subject_id)][task_desc_dict[task_selected]].values[0]
    bids_basename = make_bids_basename(
        subject=subject_id, session=None, task=task, run=None)
    bids_basename_em = make_bids_basename(
        subject=subject_id, session=None, task='EM', run=None)
    raw_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_meg.con'
    mrk_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_markers.mrk'
    elp_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_elp.elp'
    hsp_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename+'_hsp.hsp'
    raw_em_f = root_dir+'/sub-'+subject_id+'/meg/'+bids_basename_em+'_meg.con'
    print(f'read bids_basename: {bids_basename}')
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


def _read_eprime_csv(subject=None, logger=None):
    " check head_filter transformation results "
    import pandas as pd
    from devp_basicio import read_id_task_dict
    if logger is None:
        logger = Logger()
    df = read_id_task_dict()
    column = ['sub_id', '0', '1', '2']
    if subject == None:
        subject = 'sub-104'
    subject_id = subject[-3:]
    name_d = "/home/foucault/data/sourcedata/eprime_merged_proc"
    p = Path(name_d)
    i_file = p.glob(f'SUBJECT_{subject_id}/head_filter.csv')
    i_file = next(i_file)
    logger.info(f"{i_file}")
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
    return df


def reconstruct_raw(subject=None, task_selected=None):
    """Modified from epochs_filter_by_trial
    With empty_room reconstructed return for noise covariance
    """
    from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)
    from mne.preprocessing import (compute_proj_ecg, compute_proj_eog)
    from devp_basicio import define_id_list
    if subject == None:
        subject = 'sub-104'
    if task_selected == None:
        task_selected = 'spatial'
        #task_selected = 'temporal'
    subject_id = subject[-3:]
    id_list, exclude_list, data_concated_dict = define_id_list()
    raw, events, event_id, raw_em = _read_raw_bids(subject, task_selected)
    empty_room_projs = mne.compute_proj_raw(raw_em, n_mag=3)
    raw.add_proj(empty_room_projs)
    raw_em.add_proj(empty_room_projs)

    # Read eprime data
    df = _read_eprime_csv(subject)
    indexed_event_id = 253
    logger.info(f"indexed by event_id: {indexed_event_id}")
    event_index = np.where(events[:][:,2]==indexed_event_id)[0] #n=63
    task_desc_dict = dict(zip(['spatial', 'temporal'], ['Spat', 'Temp']))
    acc = df[df['ExperimentName'].str.contains(task_desc_dict[task_selected])]['Probe.ACC']
    acc_anno_index =  np.where(acc==0)[0]
    if acc_anno_index.size == 0:
        acc_anno_index = np.array([len(event_index)-1])
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
    template_file = 'template_eog_component.csv'
    template_eog_component = np.loadtxt(template_file, delimiter=',')
    # Original threshold = 0.9
    corrmap([ica], template=template_eog_component, threshold='auto', ch_type='mag',
            label='blink', plot=False)
    if hasattr(ica, "lables_"):
        ica.exclude = ica.labels_['blink']
    reconst_raw = raw.copy()
    reconst_raw.load_data().filter(l_freq=1.5, h_freq=50)
    #reconst_raw.load_data().filter(l_freq=1.5, h_freq=30)
    ica.apply(reconst_raw)
    filt_raw_em = raw_em.copy()
    filt_raw_em.load_data().filter(l_freq=1.5, h_freq=50)
    return reconst_raw, events, filt_raw_em


def reconstruct_raw_w_file(
    subject=None, task_selected=None, base_file=None, logger=None):
    """Wrapper for reconstruct_raw to save and to retrieve files
    If base_file is not assigned or files not existing,
    preprocess will computing then"""
    if logger is None:
        logger = Logger()
    if base_file:
        reconst_raw_fname = base_file.joinpath(
            f'{subject}', f'{subject}_{task_selected}_reconst_raw.fif')
        events_fname = base_file.joinpath(
            f'{subject}', f'{subject}_{task_selected}-eve.fif')
        filt_em_raw_fname = base_file.joinpath(
            f'{subject}', f'{subject}_{task_selected}_filter_em_raw.fif')
        if (reconst_raw_fname.exists() and events_fname.exists()
                and filt_em_raw_fname.exists()):
            logger.info(f'Loading {reconst_raw_fname}')
            reconst_raw = mne.io.read_raw_fif(reconst_raw_fname)
            logger.info(f'Loading {events_fname}')
            events = mne.read_events(events_fname)
            logger.info(f'Loading {filt_em_raw_fname}')
            filt_em_raw = mne.io.read_raw_fif(filt_em_raw_fname)
            return reconst_raw, events, filt_em_raw
        else:
            reconst_raw, events, filt_em_raw = reconstruct_raw(subject, task_selected)
            logger.info(f'Saving {reconst_raw_fname}')
            reconst_raw.save(reconst_raw_fname.as_posix(), overwrite=True)
            logger.info(f'Saving {events_fname}')
            mne.write_events(events_fname.as_posix(), events)
            logger.info(f'Saving {filt_em_raw_fname}')
            filt_em_raw.save(filt_em_raw_fname.as_posix(), overwrite=True)
            return reconst_raw, events, filt_em_raw
    else:
        reconst_raw, events, filt_raw_em = reconstruct_raw(subject, task_selected)
        return reconst_raw, events, filt_raw_em


def filter_raw_empty_room(subject=None, task_selected=None, logger=None):
    """Get empty room raw with the same preprocessing with data
    ICA didn't perform on empty room raw
    """
    from devp_basicio import define_id_list
    if subject == None:
        subject = 'sub-104'
    if task_selected == None:
        task_selected = 'spatial'
    logger.info(f'Loading empty room raw from subject: {subject}')
    id_list, exclude_list, data_concated_dict = define_id_list()
    raw, events, event_id, raw_em = _read_raw_bids(subject, task_selected)
    empty_room_projs = mne.compute_proj_raw(raw_em, n_mag=3)
    raw.add_proj(empty_room_projs)
    raw_em.add_proj(empty_room_projs)
    filt_raw_em = raw_em.copy()
    filt_raw_em.load_data().filter(l_freq=1.5, h_freq=50)
    return filt_raw_em


class SourceSpace(object):
    """Source space related operation"""
    def __init__(self):
        #self.logger = Logger()
        self.logger = logger
        self.SUBJECTS_DIR = Path(
            '/home/foucault/data/derivatives/working_memory/freesurfer')
        self.mne_derivatives_dname = Path(
            f'/home/foucault/data/derivatives/working_memory/mne')
        self.subject = 'sub-104'
        self.src_fname = self.mne_derivatives_dname.joinpath(
            self.subject, f'{self.subject}_ico5-src.fif')
        self.raw_fif_fname = self.mne_derivatives_dname.joinpath(
            self.subject, f'{self.subject}_task-Spat_meg.fif')
        self.bem_fname = self.SUBJECTS_DIR.joinpath(
            self.subject, "bem", f"{self.subject}-bem.fif")
        self.bem_sol_fname = self.SUBJECTS_DIR.joinpath(
            self.subject, "bem", f"{self.subject}-bem-sol.fif")

    def prepare_source_space(self, subject):
        """Prepare and save source space"""
        src = mne.setup_source_space(subject, spacing='ico5', add_dist=False,
                                     subjects_dir=self.SUBJECTS_DIR)
        self.subject = subject
        self.src_dname = Path(f'/home/foucault/data/derivatives/working_memory/mne/{subject}')
        check_dir_or_mkdir(self.src_dname)
        self.src_fname = self.src_dname.joinpath(f"{subject}_ico5-src.fif")
        self.logger.info("Saving file:")
        self.logger.info(f"{self.src_fname.as_posix()}")
        mne.write_source_spaces(self.src_fname, src, overwrite=True)

    def prepare_bem_solution(self, subject):
        """Prepare and save bem solution"""
        os.environ['FREESURFER_HOME'] = "/usr/local/freesurfer"
        self.subject = subject
        mne.bem.make_watershed_bem(subject, subjects_dir=self.SUBJECTS_DIR,
            overwrite=True, verbose='debug')
        model = mne.make_bem_model(subject, subjects_dir=self.SUBJECTS_DIR)
        self.bem_fname = self.SUBJECTS_DIR.joinpath(
            subject, "bem", f"{subject}-bem.fif")
        self.logger.info("Saving file:")
        self.logger.info(f"{self.bem_fname.as_posix()}")
        mne.write_bem_surfaces(self.bem_fname, model)
        bem_sol = mne.make_bem_solution(model)
        self.bem_sol_fname = self.SUBJECTS_DIR.joinpath(
            subject, "bem", f"{subject}-bem-sol.fif")
        self.logger.info("Saving file:")
        self.logger.info(f"{self.bem_sol_fname.as_posix()}")
        mne.write_bem_solution(self.bem_sol_fname, bem_sol)

    def convert_con2fif(self, subject=None):
        """Prepare fif to feed hsp.file"""
        task_selected = 'spatial'
        #task_selected = 'temporal'
        raw, events, event_id, raw_em = _read_raw_bids(subject, task_selected)
        self.subject = subject
        raw_fif_fname = self.mne_derivatives_dname.joinpath(
            f'{subject}', f'{subject}_task-Spat_meg.fif')
        self.raw_fif_fname = raw_fif_fname
        self.logger.info(f"Saveing file: {raw_fif_fname}")
        raw.save(raw_fif_fname)

    def prepare_trans_matrix(self, subject):
        """Prepare and save transformation matrix"""
        from mne.gui._file_traits import DigSource
        from mne.gui._fiducials_gui import MRIHeadWithFiducialsModel
        from mne.gui._coreg_gui import CoregModel
        from mne.coreg import get_mni_fiducials
        from mne.surface import dig_mri_distances
        from mne.io import write_fiducials
        from mne.io.constants import FIFF
        fids_mri = get_mni_fiducials(subject, subjects_dir=self.SUBJECTS_DIR.as_posix())
        fids_fname = self.SUBJECTS_DIR.joinpath(subject, "bem", f"{subject}-fiducials.fif")
        self.logger.info(f"Saveing file: {fids_fname}")
        write_fiducials(fids_fname, fids_mri, coord_frame=FIFF.FIFFV_COORD_MRI)
        hsp = DigSource()
        raw_fif_fname = self.mne_derivatives_dname.joinpath(
            f'{subject}', f'{subject}_task-Spat_meg.fif')
        self.logger.info(f'Reading raw_fif for hsp: {raw_fif_fname}')
        hsp.file = raw_fif_fname.as_posix()
        trans_fname = self.mne_derivatives_dname.joinpath(
            f'{subject}', f'{subject}-trans.fif')
        # Set up subject MRI source space with fiducials
        mri = MRIHeadWithFiducialsModel(
            subjects_dir=self.SUBJECTS_DIR.as_posix(), subject=subject)
        # Set up coreg model
        model = CoregModel(mri=mri, hsp=hsp)
        # Get best fit from initial coreg fit for outlier detection
        it_fids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
        errs_it = []
        for iterate in it_fids:
            model.reset()
            model.icp_fid_match = 'matched'
            # Do initial fit to fiducials
            model.fit_fiducials()
            model.icp_iterations = int(iterate)
            model.nasion_weight = 2.  # For this fit we know the nasion is not precise
            # Overweighting nasion at this step also seems to throw off the fit
            model.fit_icp()
            model.omit_hsp_points(distance=5. / 1000)  # Distance is in meters
            errs_temp = model._get_point_distance()
            if len(errs_temp) > 50:
                errs_it.append(np.median(errs_temp))
            else:
                errs_it.append(999.)
        it_fid = it_fids[np.argmin(errs_it)]
        # Test final coreg fit
        wts = [5., 10., 15.]
        it_icp = [10, 20, 30, 40, 50]
        err_icp = np.ones([len(wts), len(it_icp)])
        pts_icp = np.ones([len(wts), len(it_icp)], dtype='int64')
        for j, wt in enumerate(wts):
            for k, iterate in enumerate(it_icp):
                # Repeat best-fitting steps from above
                model.reset
                model.icp_fid_match = 'matched'
                model.fit_fiducials()
                model.icp_iterations = int(it_fid)
                model.nasion_weight = 2
                model.fit_icp()
                model.omit_hsp_points(distance=5. / 1000)
                # Test new parms
                model.nasion_weight = wt
                model.icp_iterations = int(iterate)
                model.fit_icp()
                errs_temp = model._get_point_distance()
                if len(errs_temp) > 50:
                    err_icp[j, k] = (np.median(errs_temp))
                    pts_icp[j, k] = len(errs_temp)
                else:
                    err_icp[j, k] = 999.
                    pts_icp[j, k] = int(1)
                print(err_icp[j, k])

        idx_wt, idx_it = np.where(err_icp == np.min(err_icp))
        wt = wts[idx_wt[0]]
        iterate = it_icp[idx_it[0]]
        errs_icp = np.min(err_icp)
        num_pts_icp = pts_icp[idx_wt[0], idx_it[0]]

        model.reset()
        model.icp_fid_match = 'matched'
        model.fit_fiducials()
        model.icp_iterations = int(it_fid)
        model.nasion_weight = 2.
        model.fit_icp()
        model.omit_hsp_points(distance=5. / 1000)
        model.nasion_weight = wt
        model.icp_iterations = int(iterate)
        model.fit_icp()
        logger.info(f"saveing file: {trans_fname}")
        model.save_trans(fname=trans_fname)
        errs_icp = model._get_point_distance()
        logger.info(f'Reading raw_fif: {raw_fif_fname}')
        raw = mne.io.Raw(raw_fif_fname)
        errs_nearest = dig_mri_distances(
            raw.info, trans_fname, subject, self.SUBJECTS_DIR)
        num_pts_orig = len(raw.info['dig']) - 3  # Subtract fiducials
        num_pts_icp = len(errs_icp)
        num_pts_nn = len(errs_nearest)

        self.logger.info('Median distance from digitized points to head surface is %.3f mm'
            % np.median(errs_icp*1000))
        self.logger.info('''Median distance from digitized points to head surface using nearest
        neighbor is %.3f mm''' % np.median(errs_nearest*1000))

        outerr = [num_pts_orig, it_fid, wt, iterate, num_pts_icp,
                str(1000*np.median(errs_icp)), num_pts_nn]
        self.logger.info(outerr)

    def create_high_dense_head_surf(self, subject):
        """Create high-resolution head surfaces for coordinate alignment"""
        run_command([
            "mne", "make_scalp_surfaces", "-s", subject, "-d", self.SUBJECTS_DIR,
            "--no-decimate", "--overwrite"
        ])
        self.logger.info(f"Created high-resolution head surfaces for {subject}")

    def check_coregistration(self, subject):
        """Check coregistration"""
        from mayavi import mlab
        mlab.options.offscreen = True
        raw_fif_fname = self.mne_derivatives_dname.joinpath(
            f'{subject}', f'{subject}_task-Spat_meg.fif')
        self.logger.info(f'Reading raw_fif for hsp: {raw_fif_fname}')
        raw = mne.io.Raw(raw_fif_fname)
        trans_fname = self.mne_derivatives_dname.joinpath(
            f'{subject}', f'{subject}-trans.fif')
        fig = mne.viz.plot_alignment(raw.info, trans=trans_fname, subject=subject,
                                    subjects_dir=self.SUBJECTS_DIR, surfaces='head-dense',
                                    dig=True, eeg=[], meg='sensors', show_axes=True,
                                    coord_frame='meg', mri_fiducials=True)
        mne.viz.set_3d_view(fig, 45, 90, distance=.8, focalpoint=(0., 0., 0.))
        coreg_png = self.mne_derivatives_dname.joinpath(
            f'{subject}', f'{subject}_coreg.png')
        logger.info("saving file:")
        logger.info(f"{coreg_png.as_posix()}")
        fig.scene.save_png(coreg_png.as_posix())

    def prepare_forward_solution(self, subject):
        """Prepare and save forward solution"""
        self.logger.info(f'prepare_forward_solution')
        raw_fif_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}_task-Spat_meg.fif')
        self.logger.info(f'get raw_fif: {raw_fif_fname}')
        trans_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}-trans.fif')
        self.logger.info(f'get trans_fname: {trans_fname}')
        src_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}_ico5-src.fif')
        bem_sol_fname = self.SUBJECTS_DIR.joinpath(
            subject, "bem", f"{subject}-bem-sol.fif")
        fwd = mne.make_forward_solution(
            raw_fif_fname.as_posix(), trans=trans_fname.as_posix(),
            src=src_fname.as_posix(), bem=bem_sol_fname.as_posix(),
            meg=True, eeg=False, ignore_ref=True, mindist=5.0, n_jobs=4)
        fwd_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}-fwd.fif')
        self.logger.info(f'Writing fwd_fname: {fwd_fname}')
        mne.write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=None)

    def prepare_noise_cov(self, subject):
        """Prepare noise covariance"""
        self.logger.info('Prepare noise covariance')
        raw_empty_room = filter_raw_empty_room(
            subject, logger=self.logger)
        noise_cov = mne.compute_raw_covariance(
            raw_empty_room, tmin=0, tmax=None, method='auto', picks='meg')
        cov_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}-cov.fif')
        self.logger.info(f'Writing cov_fname: {cov_fname}')
        mne.write_cov(cov_fname, noise_cov)

    def prepare_inverse_operator(self, subject=None, task_selected=None):
        """Prepare inverse operator"""
        self.logger.info('Prepare inverse operator')
        if subject == None:
            subject = 'sub-104'
        if task_selected == None:
            task_selected = 'spatial'
        raw, events, event_id, raw_em = _read_raw_bids(subject, task_selected)
        cov_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}-cov.fif')
        self.logger.info(f'Loading cov_fname: {cov_fname}')
        cov = mne.read_cov(cov_fname)
        fwd_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}-fwd.fif')
        self.logger.info(f'Loading fwd_fname: {fwd_fname}')
        fwd = mne.read_forward_solution(fwd_fname)
        inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, loose=0.2)
        inv_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}-inv.fif')
        self.logger.info(f'Writing inv_fname: {inv_fname}')
        mne.minimum_norm.write_inverse_operator(inv_fname, inv)


class SourceSpaceStat(SourceSpace):
    def __init__(self):
        super().__init__()

    def get_epochs_encoder_sti2(self, subject=None, task_selected=None):
        """Get epochs time lock to encoder stimulus 2"""
        self.logger.info('Get epochs time lock to encoder stimulus 2...')
        recon_raw, events, filt_raw_em = reconstruct_raw(subject, task_selected)
        event_ids = list(np.unique(events[:,2]))
        target_event_id, tmin, tmax = 253, -0.1, 2
        baseline = None
        logger.info(f"target event_id: {target_event_id}")
        # remove the reject=dict(mag=1.5e-12) option, too much data point lost
        epochs = mne.Epochs(recon_raw, events, target_event_id, tmin, tmax,
                            baseline=baseline, picks='meg',
                            preload=True)
        return epochs

    def prepare_sti2_baseline_noise_cov(self, subject, task_selected):
        """Prepare noise covariance"""
        self.logger.info('Prepare sti2 baseline noise covariance')
        epochs = self.get_epochs_encoder_sti2(subject, task_selected)
        noise_cov_baseline = mne.compute_covariance(epochs, tmax=0, method='auto', picks='meg')
        cov_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}_{task_selected}_sti2_baseline-cov.fif')
        self.logger.info(f'Writing cov_fname: {cov_fname}')
        mne.write_cov(cov_fname, noise_cov)

    def prepare_sti2_baseline_inverse_operator(self, subject=None, task_selected=None):
        """Prepare inverse operator"""
        self.logger.info('Prepare inverse operator')
        raw, events, event_id, raw_em = _read_raw_bids(subject, task_selected)
        cov_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}_{task_selected}_sti2_baseline-cov.fif')
        self.logger.info(f'Loading cov_fname: {cov_fname}')
        cov = mne.read_cov(cov_fname)
        fwd_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}-fwd.fif')
        self.logger.info(f'Loading fwd_fname: {fwd_fname}')
        fwd = mne.read_forward_solution(fwd_fname)
        inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, loose=0.2)
        inv_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}_{task_selected}_sti2_baseline-inv.fif')
        self.logger.info(f'Writing inv_fname: {inv_fname}')
        mne.minimum_norm.write_inverse_operator(inv_fname, inv)

    def get_contrast_epochs_sti2(self, subject=None):
        """Get contrast"""
        from mne.epochs import equalize_epoch_counts
        spat_epochs = self.get_epochs_encoder_sti2(
            subject, task_selected='spatial')
        temp_epochs = self.get_epochs_encoder_sti2(
            subject, task_selected='temporal')
        # equalize_epoch_counts([spat_epochs, temp_epochs])
        return spat_epochs.average(), temp_epochs.average()

    def get_morphed_mat(self, subject, inv):
        """Get morphed pair"""
        self.logger.info('Get morphed pair')
        # Read the source space we are morphing to
        src_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}_ico5-src.fif')
        self.logger.info(f'Loading src_fname: {src_fname}')
        src = mne.read_source_spaces(src_fname)
        fsave_vertices = [s['vertno'] for s in src]
        self.logger.info(f'fsave_vertices: {fsave_vertices}')
        morph_mat = mne.compute_source_morph(
            src=inv['src'], subject_to='fsaverage',
            spacing=fsave_vertices, subjects_dir=self.SUBJECTS_DIR).morph_mat
        n_vertices_fsave = morph_mat.shape[0]
        self.logger.info(f'n_vertices_fsave: {n_vertices_fsave}')
        return morph_mat, n_vertices_fsave

    def get_contrast_sti2(self, subject=None):
        """Get stc contrast"""
        self.logger.info('Get stc contrast')
        snr = 3.0
        lambda2 = 1.0 / snr ** 2
        method = "dSPM"  # use dSPM method (could also be MNE, sLORETA, or eLORETA)
        inv_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}-inv.fif')
        self.logger.info(f'Loading inv_fname: {inv_fname}')
        inv = mne.minimum_norm.read_inverse_operator(inv_fname)
        spat_evoked, temp_evoked = self.get_contrast_epochs_sti2(subject)
        spat_stc = mne.minimum_norm.apply_inverse(spat_evoked, inv, lambda2, method)
        temp_stc = mne.minimum_norm.apply_inverse(temp_evoked, inv, lambda2, method)
        # Only deal with t > 0, cropping to reduce multiple comparisons
        spat_stc.crop(0, None)
        temp_stc.crop(0, None)
        # tmin = spat_stc.tmin
        # tstep = spat_stc.tstep # tstep = 0.001
        n_vertices_sample, n_times = spat_stc.data.shape
        self.logger.info(f'n_vertices_sample: {n_vertices_sample}')
        morph_mat, n_vertices_fsave = self.get_morphed_mat(subject, inv)
        spat_data = morph_mat.dot(spat_stc.data)
        temp_data = morph_mat.dot(temp_stc.data)
        return np.abs(spat_data) - np.abs(temp_data)

    def stack_stc_contrast(self, subjects):
        from devp_basicio import save_list_pkl
        for subject in subjects[19:]:
            self.logger.info(f'Processing {subject} ...')
            contrast = self.get_contrast_sti2(subject)
            save_list_pkl(contrast, f'{subject}_contrast_sti2.pkl', logger=self.logger)

    def load_stc_contrast(self, subjects):
        from devp_basicio import load_list_pkl, define_id_by_aging_dict
        aging_dict = define_id_by_aging_dict()
        contrast_list = list()
        _c = 0
        for subject in subjects:
            # pick just young
            if int(subject[-3:]) not in aging_dict['young']:
                continue
            self.logger.info(f'Processing {subject} ...')
            contrast = load_list_pkl(f'{subject}_contrast_sti2.pkl', logger=self.logger)
            contrast_list.append(contrast)
            _c += 1
            # some safe threshold
            if _c == 20:
                break
        self.logger.info(f'Get #participants: {_c}')
        # XXX: check
        stack = np.stack(contrast_list, axis=0)
        # __import__('IPython').embed()
        # __import__('sys').exit()
        return stack, _c

    def sti2_spatio_temporal_cluster_1samp_test(self, subjects):
        from devp_basicio import save_list_pkl
        from scipy import stats as stats
        from mne.stats import spatio_temporal_cluster_1samp_test
        X, n_subjects = self.load_stc_contrast(subjects)
        X = np.transpose(X, [0, 2, 1])
        self.logger.info('Computing connectivity.')
        src_fname = '/home/foucault/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-ico-5-src.fif'
        src = mne.read_source_spaces(src_fname)
        connectivity = mne.spatial_src_connectivity(src)
        #p_threshold = 1.0e-2
        p_threshold = 0.1
        t_threshold = -stats.distributions.t.ppf(
            p_threshold / 2., n_subjects - 1)
        self.logger.info('Clustering.')
        T_obs, clusters, cluster_p_values, H0 = clu = \
            spatio_temporal_cluster_1samp_test(
                X, connectivity=connectivity, n_jobs=1,
                threshold=t_threshold, buffer_size=None,
                verbose=True)
        save_list_pkl(clu, 'sti2_spatio_temporal_cluster_1samp_test.pkl', logger=self.logger)
        # Now select the clusters that are sig. at p < 0.05 (note that this value
        # is multiple-comparisons corrected).
        good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

    def sti2_plot_spatio_temporal_cluster_1samp_test(self):
        from mayavi import mlab
        mlab.options.offscreen = True
        from devp_basicio import load_list_pkl
        from mne.stats import summarize_clusters_stc
        clu = load_list_pkl('sti2_spatio_temporal_cluster_1samp_test.pkl', logger=self.logger)
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
        stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep, p_thresh=0.5, vertices=fsave_vertices, subject='fsaverage')
        brain = stc_all_cluster_vis.plot(hemi='split', views='lat', subjects_dir=self.SUBJECTS_DIR, time_label='Duration significant (ms)', size=(800, 800), smoothing_steps=5)
        figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
        w_png = figures_d.joinpath("clusters.png")
        self.logger.info(f'saveing fig: {w_png.as_posix()}')
        brain.save_image(w_png.as_posix())
        __import__('IPython').embed()
        __import__('sys').exit()

    def sti2_tfce_hat_permutation_cluster_1samp_test(self, subjects):
        from devp_basicio import save_list_pkl
        from mne.stats import permutation_cluster_1samp_test, ttest_1samp_no_p
        from functools import partial
        X, n_subjects = self.load_stc_contrast(subjects)
        X = np.transpose(X, [0, 2, 1])
        self.logger.info('Computing connectivity.')
        src_fname = '/home/foucault/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-ico-5-src.fif'
        src = mne.read_source_spaces(src_fname)
        connectivity = mne.spatial_src_connectivity(src)
        #p_threshold = 1.0e-2
        # p_threshold = 0.1
        # t_threshold = -stats.distributions.t.ppf(
            # p_threshold / 2., n_subjects - 1)
        self.logger.info('Clustering.')
        sigma = 1e-3
        n_permutations = 'all'
        stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
        threshold_tfce = dict(start=0, step=0.2)
        t_tfce_hat, clusters, p_tfce_hat, H0 = clu = \
            permutation_cluster_1samp_test(
                X, n_jobs=1, threshold=threshold_tfce, connectivity=None,
                n_permutations=n_permutations, stat_fun=stat_fun_hat, buffer_size=None)
        save_list_pkl(clu, 'sti2_tfce_hat_permutation_cluster_1samp_test.pkl', logger=self.logger)
        __import__('IPython').embed()
        __import__('sys').exit()

    def sti2_tfce_permutation_cluster_1samp_test(self, subjects):
        from devp_basicio import save_list_pkl
        from mne.stats import permutation_cluster_1samp_test, ttest_1samp_no_p
        from functools import partial
        X, n_subjects = self.load_stc_contrast(subjects)
        X = np.transpose(X, [0, 2, 1])
        self.logger.info('Computing connectivity.')
        src_fname = '/home/foucault/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-ico-5-src.fif'
        src = mne.read_source_spaces(src_fname)
        connectivity = mne.spatial_src_connectivity(src)
        #p_threshold = 1.0e-2
        # p_threshold = 0.1
        # t_threshold = -stats.distributions.t.ppf(
            # p_threshold / 2., n_subjects - 1)
        self.logger.info('Clustering.')
        n_permutations = 'all'
        # sigma = 1e-3
        # stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
        threshold_tfce = dict(start=0, step=0.2)
        t_tfce_hat, clusters, p_tfce_hat, H0 = clu = \
            permutation_cluster_1samp_test(
                X, n_jobs=1, threshold=threshold_tfce, connectivity=None,
                n_permutations=n_permutations,  buffer_size=None)
        save_list_pkl(clu, 'sti2_tfce_permutation_cluster_1samp_test.pkl', logger=self.logger)
        __import__('IPython').embed()
        __import__('sys').exit()

    def sti2_permutation_cluster_1samp_test(self, subjects):
        from devp_basicio import save_list_pkl
        from mne.stats import permutation_cluster_1samp_test, ttest_1samp_no_p
        from functools import partial
        X, n_subjects = self.load_stc_contrast(subjects)
        X = np.transpose(X, [0, 2, 1])
        self.logger.info('Clustering.')
        n_permutations = 'all'
        threshold_tfce = dict(start=0, step=0.2)
        t_tfce_hat, clusters, p_tfce_hat, H0 = clu = \
            permutation_cluster_1samp_test(
                X, n_jobs=1, threshold=threshold_tfce, connectivity=None,
                n_permutations=n_permutations, buffer_size=None)
        save_list_pkl(clu, 'sti2_permutation_cluster_1samp_test.pkl', logger=self.logger)
        __import__('IPython').embed()
        __import__('sys').exit()


class SourceSpaceStat2(SourceSpaceStat):
    """Adaptively set event id as variable and experiment on use baseline
    as noise covariance matrix

    Diff to SourceSpaceStat:
        Change methond name from sti2 to event for generalization purpose
    """
    def __init__(self):
        super().__init__()
        self.task_list = ['spatial', 'temporal']

    def get_cov_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}_{kwargs["task_selected"]}'
            f'_{kwargs["target_event"]}_baseline-cov.fif').as_posix()

    def get_inv_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}_{kwargs["task_selected"]}'
            f'_{kwargs["target_event"]}_baseline-inv.fif').as_posix()

    def get_contrast_fname(self, **kwargs):
        return f'{kwargs["subject"]}_event_contrast_{kwargs["target_event"]}.pkl'

    def get_spatio_temporal_cluster_1samp_test_fname(self, **kwargs):
        return f'event_{kwargs["target_event"]}_spatio_temporal_cluster_1samp_test.pkl'

    def get_event_epochs_encoder(self, subject, task_selected, target_event_id):
        """Get epochs time lock to encoder stimulus 2"""
        self.logger.info('Get epochs time lock to encoder stimulus id {target_event_id}')
        recon_raw, events, filt_raw_em = reconstruct_raw(subject, task_selected)
        event_ids = list(np.unique(events[:,2]))
        target_event_id, tmin, tmax = target_event_id, -0.1, 2
        baseline = None
        logger.info(f"target event_id: {target_event_id}")
        # remove the reject=dict(mag=1.5e-12) option, too much data point lost
        epochs = mne.Epochs(recon_raw, events, target_event_id, tmin, tmax,
                            baseline=baseline, picks='meg',
                            preload=True)
        return epochs

    def get_event_epochs_baseline(self, subject, task_selected, target_event_id):
        """Get epochs time lock to encoder stimulus 2"""
        self.logger.info('Get epochs time lock to encoder stimulus id {target_event_id}')
        recon_raw, events, filt_raw_em = reconstruct_raw(subject, task_selected)
        event_ids = list(np.unique(events[:,2]))
        target_event_id, tmin, tmax = target_event_id, -1.5, 0
        baseline = None
        logger.info(f"target event_id: {target_event_id}")
        # remove the reject=dict(mag=1.5e-12) option, too much data point lost
        epochs = mne.Epochs(recon_raw, events, target_event_id, tmin, tmax,
                            baseline=baseline, picks='meg',
                            preload=True)
        return epochs

    def prepare_event_baseline_noise_cov(self, subject, task_selected, target_event):
        """Prepare noise covariance"""
        self.logger.info(f'Prepare {target_event} baseline noise covariance')
        epochs = self.get_epochs_encoder_sti2(subject, task_selected)
        # target_event_id = 254
        # epochs = self.get_event_epochs_encoder(subject, task_selected, target_event_id)
        noise_cov_baseline = mne.compute_covariance(epochs, tmax=0, method='auto')
        cov_fname = self.get_cov_fname(
            subject=subject,
            task_selected=task_selected,
            target_event=target_event)
        self.logger.info(f'Writing cov_fname: {cov_fname}')
        mne.write_cov(cov_fname, noise_cov_baseline)

    def prepare_event_baseline_inverse_operator(self, subject, task_selected, target_event):
        """Prepare inverse operator"""
        self.logger.info('Prepare inverse operator')
        raw, events, event_id, raw_em = _read_raw_bids(subject, task_selected)
        cov_fname = self.get_cov_fname(
            subject=subject,
            task_selected=task_selected,
            target_event=target_event)
        self.logger.info(f'Loading cov_fname: {cov_fname}')
        cov = mne.read_cov(cov_fname)
        fwd_fname = self.mne_derivatives_dname.joinpath(
            subject, f'{subject}-fwd.fif')
        self.logger.info(f'Loading fwd_fname: {fwd_fname}')
        fwd = mne.read_forward_solution(fwd_fname)
        inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, loose=0.2)
        inv_fname = self.get_inv_fname(
            subject=subject,
            task_selected=task_selected,
            target_event=target_event)
        self.logger.info(f'Writing inv_fname: {inv_fname}')
        mne.minimum_norm.write_inverse_operator(inv_fname, inv)

    def get_event_contrast(self, subject, target_event):
        """Get stc contrast"""
        self.logger.info('Get stc contrast')
        snr = 3.0
        lambda2 = 1.0 / snr ** 2
        method = "dSPM"  # use dSPM method (could also be MNE, sLORETA, or eLORETA)
        spat_evoked, temp_evoked = self.get_contrast_epochs_sti2(subject)
        evokeds = [spat_evoked, temp_evoked]
        stcs = list()
        data = list()
        for i, task_selected in enumerate(self.task_list):
            inv_fname = self.get_inv_fname(
                subject=subject,
                task_selected=task_selected,
                target_event=target_event)
            self.logger.info(f'Loading inv_fname: {inv_fname}')
            inv = mne.minimum_norm.read_inverse_operator(inv_fname)
            stcs.append(mne.minimum_norm.apply_inverse(evokeds[i], inv, lambda2, method))
            # Only deal with t > 0, cropping to reduce multiple comparisons
            stcs[-1].crop(0, None)
            # tmin = spat_stc.tmin
            # tstep = spat_stc.tstep # tstep = 0.001
            n_vertices_sample, n_times = stcs[-1].data.shape
            self.logger.info(f'{task_selected} n_vertices_sample: {n_vertices_sample}')
            morph_mat, n_vertices_fsave = self.get_morphed_mat(subject, inv)
            data.append(np.abs(morph_mat.dot(stcs[-1].data)))
        return data[0] - data[1]

    def stack_stc_event_contrast(self, subjects, target_event):
        from devp_basicio import save_list_pkl
        for subject in subjects:
            self.logger.info(f'Processing {subject} ...')
            contrast = self.get_event_contrast(subject, target_event)
            contrast_fname = self.get_contrast_fname(
                subject=subject,
                target_event=target_event)
            save_list_pkl(contrast, contrast_fname, logger=self.logger)

    def load_stc_event_contrast(self, subjects, target_event):
        from devp_basicio import load_list_pkl, define_id_by_aging_dict
        aging_dict = define_id_by_aging_dict()
        contrast_list = list()
        _c = 0
        for subject in subjects:
            # pick just young
            if int(subject[-3:]) not in aging_dict['young']:
                continue
            self.logger.info(f'Processing {subject} ...')
            contrast_fname = self.get_contrast_fname(
                subject=subject,
                target_event=target_event)
            contrast = load_list_pkl(contrast_fname, logger=self.logger)
            contrast_list.append(contrast)
            _c += 1
            # some safe threshold
            if _c == 20:
                break
        self.logger.info(f'Get #participants: {_c}')
        stack = np.stack(contrast_list, axis=0)
        # __import__('IPython').embed()
        # __import__('sys').exit()
        return stack, _c

    def event_spatio_temporal_cluster_1samp_test(self, subjects, target_event):
        from devp_basicio import save_list_pkl
        from scipy import stats as stats
        from mne.stats import spatio_temporal_cluster_1samp_test
        X, n_subjects = self.load_stc_event_contrast(subjects, target_event)
        X = np.transpose(X, [0, 2, 1])
        self.logger.info('Computing connectivity.')
        src_fname = '/home/foucault/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-ico-5-src.fif'
        src = mne.read_source_spaces(src_fname)
        connectivity = mne.spatial_src_connectivity(src)
        p_threshold = 1.0e-2
        t_threshold = -stats.distributions.t.ppf(
            p_threshold / 2., n_subjects - 1)
        self.logger.info('Clustering.')
        T_obs, clusters, cluster_p_values, H0 = clu = \
            spatio_temporal_cluster_1samp_test(
                X, connectivity=connectivity, n_jobs=1,
                threshold=t_threshold, buffer_size=None,
                verbose=True)
        event_spatio_temporal_cluster_1samp_test_fname = \
            self.get_spatio_temporal_cluster_1samp_test_fname(
                target_event=target_event)
        save_list_pkl(clu, event_spatio_temporal_cluster_1samp_test_fname, logger=self.logger)
        # Now select the clusters that are sig. at p < 0.05 (note that this value
        # is multiple-comparisons corrected).
        good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

    def event_plot_spatio_temporal_cluster_1samp_test(self):
        from mayavi import mlab
        mlab.options.offscreen = True
        from devp_basicio import load_list_pkl
        from mne.stats import summarize_clusters_stc
        event_spatio_temporal_cluster_1samp_test_fname = \
            self.get_spatio_temporal_cluster_1samp_test_fname(
                target_event='sti2')
        clu = load_list_pkl(event_spatio_temporal_cluster_1samp_test_fname, logger=self.logger)
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
        stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep, p_thresh=0.8, vertices=fsave_vertices, subject='fsaverage')
        brain = stc_all_cluster_vis.plot(hemi='split', views='lat', subjects_dir=self.SUBJECTS_DIR, time_label='Duration significant (ms)', size=(800, 800), smoothing_steps=5)
        figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
        w_png = figures_d.joinpath("clusters.png")
        self.logger.info(f'saveing fig: {w_png.as_posix()}')
        brain.save_image(w_png.as_posix())
        __import__('IPython').embed()
        __import__('sys').exit()

    def delete_intermediate_fils(self, subject, task_selected, target_event):
        """Target : cov_fname
                    inv_fname
                    contrast_fname
                    cluster_fname
        """
        cov_fname = self.get_cov_fname(
            subject=subject,
            task_selected=task_selected,
            target_event=target_event)
        Path(cov_fname).unlink(missing_ok=True)
        inv_fname = self.get_inv_fname(
            subject=subject,
            task_selected=task_selected,
            target_event=target_event)
        Path(inv_fname).unlink(missing_ok=True)
        contrast_fname = self.get_contrast_fname(
            subject=subject,
            target_event=target_event)
        Path(contrast_fname).unlink(missing_ok=True)
        cluster_fname = self.get_spatio_temporal_cluster_1samp_test_fname(
            target_event=target_event)
        Path(cluster_fname).unlink(missing_ok=True)


class SourceSpaceStat3(SourceSpaceStat2):
    """Adaptively set event id 1 as variable and experiment on use baseline
    as noise covariance matrix
    Verdict: sti1 baseline as noise covariance matrix has better results than sti2
    but still didn't get significant results
    """
    def __init__(self):
        super().__init__()

    def get_cov_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}_{kwargs["task_selected"]}'
            f'_{kwargs["target_event"]}_baseline_s1-cov.fif').as_posix()

    def get_inv_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}_{kwargs["task_selected"]}'
            f'_{kwargs["target_event"]}_baseline_s1-inv.fif').as_posix()

    def get_contrast_fname(self, **kwargs):
        return f'{kwargs["subject"]}_event_contrast_{kwargs["target_event"]}_s1.pkl'

    def get_event_permutation_cluster_1samp_test_fname(self, **kwargs):
        return f'event_{kwargs["target_event"]}_permutation_cluster_1samp_test.pkl'

    def get_test_fname(self, **kwargs):
        return self.get_event_permutation_cluster_1samp_test_fname(kwargs)

    def get_spatio_temporal_cluster_1samp_test_fname(self, **kwargs):
        return f'event_{kwargs["target_event"]}_spatio_temporal_cluster_1samp_test_s1.pkl'

    def prepare_event_baseline_noise_cov(self, subject, task_selected, target_event):
        """Prepare noise covariance"""
        self.logger.info(f'Prepare {target_event} baseline noise covariance')
        target_event_id = 254
        epochs = self.get_event_epochs_baseline(subject, task_selected, target_event_id)
        noise_cov_baseline = mne.compute_covariance(epochs, tmax=0, method='auto')
        cov_fname = self.get_cov_fname(
            subject=subject,
            task_selected=task_selected,
            target_event=target_event)
        self.logger.info(f'Writing cov_fname: {cov_fname}')
        mne.write_cov(cov_fname, noise_cov_baseline)

    def event_permutation_cluster_1samp_test(self, subjects, target_event):
        self.logger.info('Performing event_permutation_cluster_1samp_test')
        from devp_basicio import save_list_pkl
        from mne.stats import permutation_cluster_1samp_test, ttest_1samp_no_p
        from functools import partial
        X, n_subjects = self.load_stc_contrast(subjects)
        X = np.transpose(X, [0, 2, 1])
        self.logger.info('Clustering.')
        n_permutations = 'all'
        T_obs, clusters, cluster_p_values, H0 = clu = \
            permutation_cluster_1samp_test(
                X, n_jobs=1,
                n_permutations=n_permutations)
        test_fname = self.get_test_fname(target_event=target_event)
        # save_list_pkl(clu, f'event_{target_event}_permutation_cluster_1samp_test.pkl', logger=self.logger)
        save_list_pkl(clu, test_fname, logger=self.logger)

    def event_plot_permutation_cluster_1samp_test(self, target_event):
        from mayavi import mlab
        mlab.options.offscreen = True
        from devp_basicio import load_list_pkl
        from mne.stats import summarize_clusters_stc
        test_fname = self.get_test_fname(target_event=target_event)
        clu = load_list_pkl(test_fname, logger=self.logger)
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
        stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep, p_thresh=0.5, vertices=fsave_vertices, subject='fsaverage')
        brain = stc_all_cluster_vis.plot(hemi='split', views='lat', subjects_dir=self.SUBJECTS_DIR, time_label='Duration significant (ms)', size=(800, 800), smoothing_steps=5)
        figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
        w_png = figures_d.joinpath("clusters.png")
        self.logger.info(f'saveing fig: {w_png.as_posix()}')
        brain.save_image(w_png.as_posix())
        __import__('IPython').embed()
        __import__('sys').exit()


class SourceSpaceStat4(SourceSpaceStat2):
    """5D time-frequency beamforming based on LCMV
    """
    def __init__(self):
        super().__init__()

    def get_fwd_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'], f'{kwargs["subject"]}-fwd.fif').as_posix()

    def get_lcmv_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}_{kwargs["task_selected"]}'
            f'_{kwargs["target_event_id"]}_covariance-er_lcmv.pkl').as_posix()

    def get_lcmv_f4_f45_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}_{kwargs["task_selected"]}'
            f'_{kwargs["target_event_id"]}_covariance-er_f4_f45_lcmv.pkl').as_posix()

    def get_lcmv_f4_f45_spatial_temporal_contrast_fname(self, **kwargs):
        return self.mne_derivatives_dname.joinpath(
            kwargs['subject'],
            f'{kwargs["subject"]}'
            f'_{kwargs["target_event_id"]}_covariance-er_f4_f45_lcmv_contrast.pkl').as_posix()

    def get_lcmv_spatio_temporal_cluster_1samp_test_fname(self, **kwargs):
        return f'lcmv_{kwargs["freq"]}_{kwargs["target_event"]}_spatio_temporal_cluster_1samp_test.pkl'

    def get_plot_lcmv_freq_f4_f45_clusters_fname(self, **kwargs):
        figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
        return figures_d.joinpath(
            f'lcmv_{kwargs["freq"]}_{kwargs["target_event_id"]}_'
            'f4_f45_covariance_er_clusters.png').as_posix()

    def get_lcmv_event_epochs_config(self):
        tmin, tmax = -0.5, 2
        tmin_plot, tmax_plot = -0.1, 1.75
        return tmin, tmax, tmin_plot, tmax_plot

    def get_lcmv_event_epochs_encoder(self, subject, task_selected, target_event_id):
        """Get epochs time lock to encoder stimulus 2"""
        self.logger.info(f'Get epochs time lock to encoder stimulus id {target_event_id}')
        recon_raw, events, filt_raw_em = reconstruct_raw(subject, task_selected)
        event_ids = list(np.unique(events[:,2]))
        # fine-tune time length
        tmin, tmax, tmin_plot, tmax_plot = self.get_lcmv_event_epochs_config()
        baseline = None
        logger.info(f"target event_id: {target_event_id}")
        # remove the reject=dict(mag=1.5e-12) option, too much data point lost
        epochs = mne.Epochs(recon_raw, events, target_event_id, tmin, tmax,
                            baseline=baseline, picks='meg',
                            preload=False)
        return epochs

    def get_lcmv_noise_cov(self, subject, target_event_id):
        """Prepare empty room noise covariance used in lcmv method"""
        from mne.event import make_fixed_length_events
        self.logger.info('Prepare empty room noise covariance used in lcmv method')
        raw_empty_room = filter_raw_empty_room(
            subject, logger=self.logger)
        # Create artificial events for empty room noise data
        events_noise = make_fixed_length_events(raw_empty_room, target_event_id, duration=1.)
        tmin, tmax, tmin_plot, tmax_plot = self.get_lcmv_event_epochs_config()
        epochs_noise = mne.Epochs(raw_empty_room, events_noise, target_event_id, tmin, tmax)
        return epochs_noise, raw_empty_room

    def get_lcmv_solution(self, subject, task_selected, target_event_id):
        # Make sure the number of noise epochs is the same as data epochs
        tmin, tmax, tmin_plot, tmax_plot = self.get_lcmv_event_epochs_config()
        epochs = \
            self.get_lcmv_event_epochs_encoder(subject,
                                               task_selected,
                                               target_event_id)
        epochs_noise, raw_noise = self.get_lcmv_noise_cov(subject, target_event_id)
        epochs_noise = epochs_noise[:len(epochs.events)]
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
        # corresponding to get_lcmv_f4_f45_fname
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
        for (l_freq, h_freq) in freq_bins:
            raw_band = raw_noise.copy()
            raw_band.filter(l_freq, h_freq, n_jobs=1, fir_design='firwin')
            epochs_band = mne.Epochs(raw_band, epochs_noise.events, target_event_id,
                                     tmin=tmin_plot, tmax=tmax_plot, baseline=None,
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

    def save_lcmv_solution(self, subject, task_selected, target_event_id):
        from devp_basicio import save_list_pkl
        stcs = self.get_lcmv_solution(subject, task_selected, target_event_id)
        # lcmv_fname = self.get_lcmv_fname(
            # subject=subject, task_selected=task_selected,
            # target_event_id=target_event_id)
        lcmv_fname = self.get_lcmv_f4_f45_fname(
            subject=subject, task_selected=task_selected,
            target_event_id=target_event_id)
        save_list_pkl(stcs, lcmv_fname, logger=self.logger)

    def get_lcmv_spatial_temporal_contrast(self, subject, target_event_id):
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
            # lcmv_fname = self.get_lcmv_fname(
                # subject=subject, task_selected=task_selected,
                # target_event_id=target_event_id)
            lcmv_fname = self.get_lcmv_f4_f45_fname(
                subject=subject, task_selected=task_selected,
                target_event_id=target_event_id)
            stcs = load_list_pkl(lcmv_fname, logger=self.logger)
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

    def save_lcmv_spatial_temporal_contrast(self, subject, target_event_id):
        from devp_basicio import save_list_pkl
        freq_stcs_contrast_dict = \
            self.get_lcmv_spatial_temporal_contrast(subject, target_event_id)
        contrast_fname = self.get_lcmv_f4_f45_spatial_temporal_contrast_fname(
            subject=subject,
            target_event_id=target_event_id)
        save_list_pkl(freq_stcs_contrast_dict,
                      contrast_fname, logger=self.logger)

    def load_lcmv_spatial_temporal_contrast(self, subject, target_event_id):
        from devp_basicio import load_list_pkl
        contrast_fname = \
        self.get_lcmv_f4_f45_spatial_temporal_contrast_fname(
            subject=subject,
            target_event_id=target_event_id)
        freq_stcs_contrast_dict = load_list_pkl(
            contrast_fname, logger=self.logger)
        return freq_stcs_contrast_dict

    def load_stacking_lcmv_spatial_temporal_contrast(self, subjects, target_event_id):
        """Target_event_id: 253"""
        from devp_basicio import load_list_pkl
        _list = list()
        for subject in subjects:
            self.logger.info(f'Processing {subject} ...')
            freq_stcs_contrast_dict = \
            self.load_lcmv_spatial_temporal_contrast(
                subject, target_event_id)
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

    def lcmv_spatio_temporal_cluster_1samp_test(self, subjects, target_event):
        """Target_event: sti2"""
        print("test")
        from devp_basicio import save_list_pkl
        from scipy import stats as stats
        from mne.stats import spatio_temporal_cluster_1samp_test
        _list, n_subjects = self.load_stacking_lcmv_spatial_temporal_contrast(subjects, target_event)
        self.logger.info('Computing connectivity.')
        src_fname = '/home/foucault/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-ico-5-src.fif'
        src = mne.read_source_spaces(src_fname)
        connectivity = mne.spatial_src_connectivity(src)
        p_threshold = 1.0e-1
        t_threshold = -stats.distributions.t.ppf(
            p_threshold / 2., n_subjects - 1)
        freq_iter = _list[0].keys()
        freq_dict = self._transform_to_training(_list)
        for freq in freq_iter:
            X = freq_dict[freq]
            X = np.transpose(X, [0, 2, 1])
            self.logger.info(f'Clustering in band: {freq}')
            T_obs, clusters, cluster_p_values, H0 = clu = \
                spatio_temporal_cluster_1samp_test(
                    X, connectivity=connectivity, n_jobs=1,
                    threshold=t_threshold, buffer_size=None,
                    verbose=True)
            lcmv_spatio_temporal_cluster_1samp_test_fname = \
                self.get_lcmv_spatio_temporal_cluster_1samp_test_fname(
                    target_event=target_event, freq=freq)
            save_list_pkl(clu, lcmv_spatio_temporal_cluster_1samp_test_fname, logger=self.logger)
            # Now select the clusters that are sig. at p < 0.05 (note that this value
            # is multiple-comparisons corrected).
            good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

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
            w_png = self.get_plot_lcmv_freq_f4_f45_clusters_fname(
                freq=freq, target_event_id=target_event_id)
            self.logger.info(f'saveing fig: {w_png}')
            brain.save_image(w_png)
            # __import__('IPython').embed()
            # __import__('sys').exit()


def devp_source_space():
    """ prototype """
    #import matplotlib.pyplot as plt
    from mne.gui._file_traits import DigSource
    from mne.gui._fiducials_gui import MRIHeadWithFiducialsModel
    from mne.gui._coreg_gui import CoregModel
    from mne.coreg import get_mni_fiducials
    from mne.surface import dig_mri_distances
    from mne.io import write_fiducials
    from mne.io.constants import FIFF
    from mayavi import mlab
    mlab.options.offscreen = True
    # SUBJECTS_DIR = "/usr/local/freesurfer/subjects"
    # subject = "/home/foucault/data/derivatives/working_memory/freesurfer/sub-103"
    SUBJECTS_DIR = Path("/home/foucault/data/derivatives/working_memory/freesurfer")
    #subject = "sub-103"
    subject = "sub-104"
    src_dname = Path("/home/foucault/data/derivatives/working_memory/intermediate/tmp")
    src_fname = src_dname.joinpath("test_ico5-src.fif")
    # src = mne.setup_source_space(subject, spacing='ico5', add_dist=False,
                                 # subjects_dir=SUBJECTS_DIR)
    # logger.info("saving file:")
    # logger.info(f"{src_fname.as_posix()}")
    # mne.write_source_spaces(src_fname, src, overwrite=True)
    # __import__('IPython').embed()
    # __import__('sys').exit()
    # src = mne.read_source_spaces(fname_src)
    # fig = src.plot(subjects_dir=SUBJECTS_DIR)
    #fig, axes = plt.subplots(1, 1, figsize=(10, 7), sharex=True, sharey=True)
    figures_d = Path("/home/foucault/data/derivatives/working_memory/intermediate/figure")
    w_png = figures_d.joinpath(f"test.png")
    # fig.scene.save_png(w_png.as_posix())
    # call freesurfer environment
    # mne.bem.make_watershed_bem(subject, subjects_dir=SUBJECTS_DIR.as_posix(),
        # overwrite=True, verbose='debug')

    # fig = mne.viz.plot_bem(subject=subject, subjects_dir=SUBJECTS_DIR,
        # brain_surfaces='white', src=src, orientation='coronal', show=False)
    # logger.info("saving file:")
    # logger.info(f"{w_png.as_posix()}")
    # fig.savefig(w_png.as_posix(), dpi=300)

    # model = mne.make_bem_model(subject, subjects_dir=SUBJECTS_DIR.as_posix())
    # bem_fname = SUBJECTS_DIR.joinpath(subject, "bem", f"{subject}-bem.fif")
    # mne.write_bem_surfaces(bem_fname, model)
    # bem_sol = mne.make_bem_solution(model)
    bem_sol_fname = SUBJECTS_DIR.joinpath(subject, "bem", f"{subject}-bem-sol.fif")
    # mne.write_bem_solution(bem_sol_fname, bem_sol)
    raw_fname = f"/home/foucault/data/rawdata/working_memory/sub-104/meg/{subject}_task-Spat_meg.fif"
    trans_fname = Path("/home/foucault/data/derivatives/working_memory/mne").joinpath(f"{subject}-trans.fif")
    # fwd = mne.make_forward_solution(raw_fname, trans=trans_fname, src=src_fname,
                                    # bem=bem_sol_fname.as_posix(), meg=True, eeg=False,
                                    # ignore_ref=True, mindist=5.0, n_jobs=4)
    fwd_fname = src_dname.joinpath("test-fwd.fif")
    # mne.write_forward_solution(fwd_fname.as_posix(), fwd, overwrite=False, verbose=None)
    fwd = mne.read_forward_solution(fwd_fname)
    # logger.info("saving file:")
    # logger.info(f"{fwd_fname.as_posix()}")
    # leadfield = fwd['sol']['data']
    # print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
    # fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                             # use_cps=True)
    # leadfield = fwd_fixed['sol']['data']
    # print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
    reconst_raw, events = epochs_filter_by_trial(sub_index=0, task_selected=None)
    event_ids = list(np.unique(events[:,2]))
    #indexed_event_id = 253
    target_event_id, tmin, tmax = 253, -0.1, 2
    baseline = None
    # logger.info(f"indexed by event_id: {indexed_event_id}")
    logger.info(f"target event_id: {target_event_id}")
    event_index = np.where(events[:][:,2]==target_event_id)[0] # n=63
    # epochs = mne.Epochs(reconst_raw, events, event_ids,
    # reject=dict(mag=4000e-13), picks='meg')
    # evoked = epochs[indexed_event_id].average()
    epochs = mne.Epochs(reconst_raw, events, target_event_id, tmin, tmax,
                        baseline=baseline, picks='meg',
                        reject=dict(mag=1.5e-12),
                        preload=True)
    evoked = epochs.average()
    #evoked = epochs[target_event_id].average()
    cov = mne.compute_covariance(epochs, tmax=0., method='auto')
    inv = mne.minimum_norm.make_inverse_operator(reconst_raw.info, fwd, cov, loose=0.2)
    inv_fname = Path("/home/foucault/data/derivatives/working_memory/mne").joinpath(f"{subject}-inv.fif")
    mne.minimum_norm.write_inverse_operator(inv_fname, inv)
    stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2=1. / 9.)
    #def devp_
    # peak_vertex, peak_time = stc.get_peak(hemi='lh', vert_as_index=True,
                                          # time_as_index=True)
    peak_vertex_surf, peak_time = stc.get_peak(hemi='lh')
    # peak_time = 172
    # peak_vertex_surf = 48268
    #  array([ 19.5993557 , -21.20473099, -40.02994537])
    #brain.data['surfaces'][0
    #peak_vertex_surf = 3580
    #peak_time = 424
    #peak_vertex_surf = stc.lh_vertno[peak_vertex]
    # brain = stc.plot(initial_time=peak_time*1e-3, subject=subject, subjects_dir=SUBJECTS_DIR, hemi='split', views='lat')
    brain = stc.plot(initial_time=peak_time, subject=subject, subjects_dir=SUBJECTS_DIR, hemi='split', views='med')
    #brain.add_foci(peak_vertex_surf, coords_as_verts=True, hemi='lh', color='blue', scale_factor=1, alpha=1)
    #brain.add_foci(peak_vertex_surf, coords_as_verts=True, hemi='lh', color='blue', scale_factor=1, alpha=1)
    brain.add_foci(peak_vertex_surf, coords_as_verts=True, hemi='lh', color='green', scale_factor=1, alpha=1)
    brain.save_image(w_png, mode="rgba", antialiased=True)
    __import__('IPython').embed()
    __import__('sys').exit()
    # fname_fids = SUBJECTS_DIR.joinpath(subject, "bem", f"{subject}-fiducials.fif")
    # fids_mri = get_mni_fiducials(subject, subjects_dir=SUBJECTS_DIR.as_posix())
    # write_fiducials(fname_fids, fids_mri, coord_frame=FIFF.FIFFV_COORD_MRI)
    sub_index = 0
    task_selected = 'spatial'
    #task_selected = 'temporal'
    # raw, events, event_id, raw_em = _read_raw_bids(sub_index, task_selected)
    raw_fname = f"/home/foucault/data/rawdata/working_memory/sub-104/meg/{subject}_task-Spat_meg.fif"
    # logger.info(f"saveing file: {tmp_fname}")
    # raw.save(tmp_fname)

    # run_command([
        # "mne", "make_scalp_surfaces", "-s", subject, "-d", SUBJECTS_DIR,
        # "--no-decimate", "--overwrite"
    # ])
    # print(f"Created high-resolution head surfaces for {subject}")
    # raw = mne.io.read_raw_fif(raw_fname)

    hsp = DigSource()
    #hsp.file = raw.filenames[0]
    hsp.file = raw_fname
    trans_fname = Path("/home/foucault/data/derivatives/working_memory/mne").joinpath(f"{subject}-trans.fif")
    # Set up subject MRI source space with fiducials
    mri = MRIHeadWithFiducialsModel(subjects_dir=SUBJECTS_DIR.as_posix(), subject=subject)

    # Set up coreg model
    model = CoregModel(mri=mri, hsp=hsp)
    # Get best fit from initial coreg fit for outlier detection
    it_fids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
    errs_it = []
    for iterate in it_fids:
        model.reset()
        model.icp_fid_match = 'matched'
        # Do initial fit to fiducials
        model.fit_fiducials()
        model.icp_iterations = int(iterate)
        model.nasion_weight = 2.  # For this fit we know the nasion is not precise
        # Overweighting nasion at this step also seems to throw off the fit
        model.fit_icp()
        model.omit_hsp_points(distance=5. / 1000)  # Distance is in meters
        errs_temp = model._get_point_distance()
        if len(errs_temp) > 50:
            errs_it.append(np.median(errs_temp))
        else:
            errs_it.append(999.)

    it_fid = it_fids[np.argmin(errs_it)]

    # Test final coreg fit
    wts = [5., 10., 15.]
    it_icp = [10, 20, 30, 40, 50]
    err_icp = np.ones([len(wts), len(it_icp)])
    pts_icp = np.ones([len(wts), len(it_icp)], dtype='int64')
    for j, wt in enumerate(wts):
        for k, iterate in enumerate(it_icp):
            # Repeat best-fitting steps from above
            model.reset
            model.icp_fid_match = 'matched'
            model.fit_fiducials()
            model.icp_iterations = int(it_fid)
            model.nasion_weight = 2
            model.fit_icp()
            model.omit_hsp_points(distance=5. / 1000)
            # Test new parms
            model.nasion_weight = wt
            model.icp_iterations = int(iterate)
            model.fit_icp()
            errs_temp = model._get_point_distance()
            if len(errs_temp) > 50:
                err_icp[j, k] = (np.median(errs_temp))
                pts_icp[j, k] = len(errs_temp)
            else:
                err_icp[j, k] = 999.
                pts_icp[j, k] = int(1)
            print(err_icp[j, k])

    idx_wt, idx_it = np.where(err_icp == np.min(err_icp))
    wt = wts[idx_wt[0]]
    iterate = it_icp[idx_it[0]]
    errs_icp = np.min(err_icp)
    num_pts_icp = pts_icp[idx_wt[0], idx_it[0]]

    model.reset()
    model.icp_fid_match = 'matched'
    model.fit_fiducials()
    model.icp_iterations = int(it_fid)
    model.nasion_weight = 2.
    model.fit_icp()
    model.omit_hsp_points(distance=5. / 1000)
    model.nasion_weight = wt
    model.icp_iterations = int(iterate)
    model.fit_icp()
    logger.info(f"saveing file: {trans_fname}")
    model.save_trans(fname=trans_fname)
    errs_icp = model._get_point_distance()
    raw = mne.io.Raw(raw_fname)
    errs_nearest = dig_mri_distances(raw.info, trans_fname, subject, SUBJECTS_DIR)
    num_pts_orig = len(raw.info['dig']) - 3  # Subtract fiducials
    num_pts_icp = len(errs_icp)
    num_pts_nn = len(errs_nearest)

    print('Median distance from digitized points to head surface is %.3f mm'
        % np.median(errs_icp*1000))
    print('''Median distance from digitized points to head surface using nearest
    neighbor is %.3f mm''' % np.median(errs_nearest*1000))

    outerr = [num_pts_orig, it_fid, wt, iterate, num_pts_icp,
              str(1000*np.median(errs_icp)), num_pts_nn]
    print(outerr)

    # #model.hsp_weight = 30
    # # Do initial fit to fiducials
    # model.fit_fiducials()
    # # Do initial coreg fit for outlier detection
    # model.icp_iterations = int(50)
    # model.nasion_weight = 2.  # For this fit we know the nasion is not precise
    # # Overweighting at this step also seems to throw off the fit for some datasets
    # model.icp_fid_match = 'matched'
    # model.fit_icp()
    # print(model.status_text)
    # model.fit_icp()
    # print(model.status_text)
    # model.omit_hsp_points(distance=3/1000)  # Distance is in meters
    # # Do final coreg fit
    # model.nasion_weight = 5.
    # model.icp_iterations = int(50)
    # model.fit_icp()
    # logger.info(f"saveing file: {trans_fname}")
    # model.save_trans(fname=trans_fname)
    # errs_icp = model._get_point_distance()
    # raw = mne.io.Raw(raw_fname)
    # errs_nearest = dig_mri_distances(raw.info, trans_fname, subject, SUBJECTS_DIR)
    # fig = mne.viz.plot_alignment(raw.info, trans=trans_fname, subject=subject,
                                # subjects_dir=SUBJECTS_DIR, surfaces='head-dense',
                                # dig=True, eeg=[], meg='sensors', show_axes=True,
                                # coord_frame='meg', mri_fiducials=True)
    fig = mne.viz.plot_alignment(raw.info, trans=trans_fname, subject=subject,
                                subjects_dir=SUBJECTS_DIR, surfaces='head-dense',
                                dig=True, eeg=[], meg='sensors', show_axes=True,
                                coord_frame='meg', mri_fiducials=True)
    mne.viz.set_3d_view(fig, 45, 90, distance=.8, focalpoint=(0., 0., 0.))

    logger.info("saving file:")
    logger.info(f"{w_png.as_posix()}")
    fig.scene.save_png(w_png.as_posix())
    # print('Median distance from digitized points to head surface is %.3f mm'
        # % np.median(errs_icp*1000))
    # print('''Median distance from digitized points to head surface using nearest
    # neighbor is %.3f mm''' % np.median(errs_nearest*1000))
    # __import__('IPython').embed()
    # __import__('sys').exit()
    #os.environ['FREESURFER_HOME'] = "/usr/local/freesurfer"
    # fig.savefig(w_png.as_posix(), dpi=300)
    # fwd = mne.read_forward_solution(fname_fwd)
    # cov = mne.read_cov(fname_cov)


def process_subject_head(subject_id):
    subject_mri_dir = op.join(mri_dir, subject_id)
    subject = map_subjects[subject_id]


def check_hsp(sub_index=None, task_selected=None):
    from devp_basicio import define_id_list
    if sub_index == None:
        sub_index = 0
    if task_selected == None:
        #task_selected = 'spatial'
        task_selected = 'temporal'
    id_list, exclude_list, data_concated_dict = define_id_list()
    if str(id_list[sub_index]) in data_concated_dict.keys():
        raw, events, event_id, raw_em =\
            data_concated_dict[str(id_list[sub_index])](task_selected)
    else:
        raw, events, event_id, raw_em = _read_raw_bids(sub_index, task_selected)
    __import__('IPython').embed()
    __import__('sys').exit()


def main_source_space_exp_1():
    """Test empty as noise covariance"""
    sss = SourceSpaceStat()
    subjects = get_subjects_list()
    for subject in subjects:
        logger.info(f'Processing {subject} ...')
        ss.prepare_source_space(subject)
        ss.prepare_bem_solution(subject)
        ss.convert_con2fif(subject)
        ss.prepare_trans_matrix(subject)
        ss.create_high_dense_head_surf(subject)
        ss.check_coregistration(subject)
        ss.prepare_forward_solution(subject)
        ss.prepare_noise_cov(subject)
        ss.prepare_inverse_operator(subject)
        sss.get_contrast_sti2(subject)
    sss.stack_stc_contrast(subjects)
    sss.load_stc_contrast(subjects)
    sss.sti2_spatio_temporal_cluster_1samp_test(subjects)
    sss.sti2_plot_spatio_temporal_cluster_1samp_test()


def main_source_space_exp_2():
    """Test baseline covariance"""
    sss = SourceSpaceStat2()
    subjects = get_subjects_list()
    for subject in subjects:
        logger.info(f'Processing {subject} ...')
        for task_selected in sss.task_list:
            sss.prepare_event_baseline_noise_cov(subject, task_selected, 'sti2')
            sss.prepare_event_baseline_inverse_operator(subject, task_selected, 'sti2')
    sss.stack_stc_event_contrast(subjects, 'sti2')
    sss.event_spatio_temporal_cluster_1samp_test(subjects, 'sti2')


def main_source_space_exp_3():
    """Test sti1 baseline covariance"""
    sss = SourceSpaceStat3()
    subjects = get_subjects_list()
    for subject in subjects:
        logger.info(f'Processing {subject} ...')
        for task_selected in sss.task_list:
            sss.prepare_event_baseline_noise_cov(subject, task_selected, 'sti2')
            sss.prepare_event_baseline_inverse_operator(subject, task_selected, 'sti2')
    sss.stack_stc_event_contrast(subjects, 'sti2')
    def statstical_tests():
        # sss.event_spatio_temporal_cluster_1samp_test(subjects, 'sti2')
        # not significant
        # sss.event_permutation_cluster_1samp_test(subjects, 'sti2')
        # memory limitation
        pass
    statstical_tests()
    # sss.event_plot_spatio_temporal_cluster_1samp_test()


def main_source_space_exp_4():
    """Lcmv with covariance from empty room"""
    from devp_basicio import define_id_by_aging_dict
    aging_dict = define_id_by_aging_dict()
    sss = SourceSpaceStat4()
    subjects = get_subjects_list()
    # for subject in subjects:
        # logger.info(f'Processing {subject} ...')
        # for task_selected in sss.task_list:
            # sss.save_lcmv_solution(subject, task_selected, 253)
        # sss.stack_lcmv_solution(subjects)
        # sss.get_lcmv_spatial_temporal_contrast(subject, 253)
        # sss.save_lcmv_spatial_temporal_contrast(subject, 253)
    _subjects = list()
    for subject in subjects:
            # pick just young
        if int(subject[-3:]) not in aging_dict['young']:
            logger.info(f'Filtering out {subject} ...')
            continue
        _subjects.append(subject)
        logger.info(f'Appending {subject} ...')
    subjects = _subjects
    # sss.lcmv_spatio_temporal_cluster_1samp_test(subjects, 253)
    sss.plot_lcmv_spatio_temporal_cluster_1samp_test(253)


def main_source_space():
    #ss = SourceSpace()
    # sss = SourceSpaceStat()
    sss = SourceSpaceStat2()
    from mne.parallel import parallel_func
    subjects = get_subjects_list()
    #subject = f'sub-{id_list[0]}'
    n_jobs = 6
    # __import__('IPython').embed()
    # __import__('sys').exit()
    # for subject in subjects[24:]:
    # for subject in subjects:
        # logger.info(f'Processing {subject} ...')
        # ss.prepare_source_space(subject)
        # ss.prepare_bem_solution(subject)
        # ss.convert_con2fif(subject)
        # ss.prepare_trans_matrix(subject)
        # ss.create_high_dense_head_surf(subject)
        # ss.check_coregistration(subject)
        # ss.prepare_forward_solution(subject)
        # ss.prepare_noise_cov(subject)
        # ss.prepare_inverse_operator(subject)
        # sss.get_contrast_sti2(subject)
        # for task_selected in sss.task_list:
            # sss.prepare_event_baseline_noise_cov(subject, task_selected, 'sti2')
            # sss.prepare_event_baseline_inverse_operator(subject, task_selected, 'sti2')
    # sss.stack_stc_contrast(subjects)
    # sss.load_stc_contrast(subjects)
    # sss.sti2_spatio_temporal_cluster_1samp_test(subjects)
    # sss.sti2_plot_spatio_temporal_cluster_1samp_test()
    # sss.sti2_permutation_cluster_1samp_test(subjects)
    # sss.sti2_tfce_hat_permutation_cluster_1samp_test(subjects)
    # sss.sti2_tfce_permutation_cluster_1samp_test(subjects)
    # sss.stack_stc_event_contrast(subjects, 'sti2')
    # sss.event_spatio_temporal_cluster_1samp_test(subjects, 'sti2')
    sss.event_plot_spatio_temporal_cluster_1samp_test()
    # parallel, run_func, _ = parallel_func(ss.prepare_source_space, n_jobs=n_jobs)
    # parallel(run_func(subject) for subject in subjects)


def main():
    # devp_source_space()
    # main_source_space()
    # main_source_space_exp_3()
    # main_source_space_exp_4()
    ss = SourceSpace()
    base_file = ss.mne_derivatives_dname
    reconstruct_raw_w_file(
        subject='sub-104', task_selected='spatial',
        base_file=base_file, logger=ss.logger)


if __name__ == '__main__':
    main()
