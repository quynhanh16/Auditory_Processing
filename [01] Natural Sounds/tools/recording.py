import copy
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
import warnings
from pathlib import PosixPath, WindowsPath

import numpy as np
import pandas as pd
import requests

from tools.signal import (
    SignalBase,
    RasterizedSignal,
    list_signals,
    load_signal,
    load_signal_from_streams,
)
from tools.uri import local_uri, http_uri, targz_uri, NumpyEncoder, json_numpy_obj_hook

log = logging.getLogger(__name__)


class Recording:

    def __init__(self, signals, meta=None, name=None, signal_views=None):
        """
        Signals argument should be a dictionary of signal objects.
        """
        if signal_views is not None:
            self.signal_views = signal_views
            self.signals = signal_views[0]
        else:
            self.signals = signals
            self.signal_views = [signals]
        self.view_idx = 0

        # Verify that all signals are from the same recording
        recordings = [s.recording for s in self.signals.values()]
        if not recordings:
            raise ValueError("A recording must contain at least 1 signal")
        if not len(set(recordings)) == 1:
            raise ValueError("Not all signals are from the same recording.")
        if name is None:
            if len(recordings):
                self.name = recordings[0]
            else:
                self.name = "recording"
        else:
            self.name = name

        self.uri = None  # This will be lost on copying

        if meta is not None:
            self.meta = meta
        else:
            self.meta = {}

    def copy(self):
        """
        Returns a copy of this recording.
        """
        signal_views = [s.copy() for s in self.signal_views]
        other = Recording(signal_views[self.view_idx], meta=self.meta.copy())
        other.signal_views = signal_views
        other.view_idx = self.view_idx
        other.signals = signal_views[other.view_idx]

        for k, v in vars(self).items():
            if k in ["signals", "signal_views", "view_idx"]:
                continue
            setattr(other, k, copy.copy(v))
        return other

    @property
    def epochs(self):
        """
        The epochs of a recording is the superset of all signal epochs.
        """
        # Merge the epochs. Be sure to ignore index since it's just a standard
        # sequential index for each signal's epoch (e.g., index 1 in signal1 has
        # no special meaning compared to index 1 in signal2). Drop all
        # duplicates since we sometimes replicate epochs across signals and
        # return the sorted values.
        epoch_set = [s.epochs for s in self.signals.values()]
        df = pd.concat(epoch_set, ignore_index=True)
        df.drop_duplicates(inplace=True)
        df.sort_values("start", inplace=True)
        df.index = np.arange(len(df))
        return df

    # Defining __getitem__ and __setitem__ make recording objects behave
    # like dictionaries when subscripted. e.g. recording['signal_name']
    # instead of recording.get_signal('signal_name').
    # See: https://docs.python.org/3/reference/datamodel.html?emulating-container-types#emulating-container-types

    def __getitem__(self, key):
        if type(key) is int:
            return self.signal_views[key]
        else:
            return self.get_signal(key)

    def __setitem__(self, key, val):
        val.name = key
        self.add_signal(val)

    def set_view(self, view_idx=0):
        """choose a different view, typically a different masking for jackknifing.
        returns a shallow copy of the recording, signals preserved in place"""
        rec = self.copy()
        rec.signals = rec.signal_views[view_idx]
        rec.view_idx = view_idx

        return rec

    def view_subset(self, view_range):
        """
        shallow copy recording, preserving a subset of views
        view_range - list of view ids to keep
        """
        rec = self.copy()
        rec.signal_views = [rec.signal_views[v] for v in view_range]
        rec.signals = rec.signal_views[0]
        rec.view_idx = 0

        return rec

    def views(self, view_range=None):
        rec = self.copy()

        if view_range is not None:
            if type(view_range) is int:
                rec.signal_views = [rec.signal_views[view_range]]
            else:
                rec.signal_views = [rec.signal_views[v] for v in view_range]

        """return a list of all views of this recording"""
        return [rec.set_view(i) for i in range(rec.view_count)]

    @property
    def view_count(self):
        """return how many views exist in this recording"""
        return len(self.signal_views)

    def tile_views(self, view_count=1):
        """repeat current signals dict view_count times in self.signal views
        returns a shallow copy of the recording, signals preserved in place"""
        rec = self.copy()

        # rec.signal_views = [rec.signals] * view_count
        rec.signal_views = rec.signal_views * view_count
        rec.view_idx = 0
        rec.signals = rec.signal_views[rec.view_idx]

        return rec

    @staticmethod
    def load(uri):
        """
        DEPRECATED??? REPLACED by regular functions?

        Loads from a local .tgz file, a local directory, from s3,
        or from an HTTP URL containing a .tgz file. Examples:

        # Load all signals in the gus016c-a2 directory
        rec = Recording.load('/home/myuser/gus016c-a2')
        rec = Recording.load('file:///home/myuser/gus016c-a2')

        # Load the local tar gz directory.
        rec = Recording.load('file:///home/myuser/gus016c-a2.tgz')

        # Load a tgz file served from a flat filesystem
        rec = Recording.load('http://potoroo/recordings/gus016c-a2.tgz')

        # Load a tgz file created by the nems-baphy interafce
        rec = Recording.load('http://potoroo/baphy/271/gus016c-a2')

        # Load from S3:
        rec = Recording.load('s3://nems0.lbhb... TODO')
        """
        if local_uri(uri):
            if targz_uri(uri):
                rec = Recording.load_targz(local_uri(uri))
            else:
                rec = Recording.load_dir(local_uri(uri))
        elif http_uri(uri):
            rec = Recording.load_url(http_uri(uri))
        elif uri[0:6] == "s3://":
            raise NotImplementedError
        else:
            raise ValueError("Invalid URI: {}".format(uri))
        rec.uri = uri
        return rec

    @staticmethod
    def load_dir(directory_or_targz):
        """
        Loads all the signals (CSV/JSON pairs) found in DIRECTORY or
        .tgz file, and returns a Recording object containing all of them.
        DEPRECATED???
        """
        if os.path.isdir(directory_or_targz):
            files = list_signals(directory_or_targz)
            basepaths = [os.path.join(directory_or_targz, f) for f in files]
            signals = [load_signal(f) for f in basepaths]
            signals_dict = {s.name: s for s in signals}
            return Recording(signals=signals_dict)
        else:
            m = "Not a directory: {}".format(directory_or_targz)
            raise ValueError(m)

    @staticmethod
    def load_targz(targz):
        """
        Loads the recording object from a tgz file.
        DEPRECATED???
        """
        if os.path.exists(targz):
            with open(targz, "rb") as stream:
                return load_recording_from_targz_stream(stream)
        else:
            m = "Not a .tgz file: {}".format(targz)
            raise ValueError(m)

    @staticmethod
    def load_url(url):
        """
        Loads the recording object from a URL. File must be tgz format.
        DEPRECATED???
        """
        r = requests.get(url, stream=True)
        if not (
                r.status_code == 200
                and (
                        r.headers["content-type"] == "application/gzip"
                        or r.headers["content-type"] == "text/plain"
                        or r.headers["content-type"] == "application/x-gzip"
                        or r.headers["content-type"] == "application/x-compressed"
                        or r.headers["content-type"] == "application/x-tar"
                        or r.headers["content-type"] == "application/x-tgz"
                )
        ):
            log.info("got response: %s, %d", r.headers, r.status_code)
            m = "Error loading URL: {}".format(url)
            log.error(m)
            raise Exception(m)
        obj = io.BytesIO(r.raw.read())  # Not sure why I need this!
        return load_recording_from_targz_stream(obj)

    def save_dir(self, directory, fmt=None):
        """
        Saves all the signals (CSV/JSON pairs) in this recording into
        DIRECTORY in a new directory named the same as this recording.
        """
        # SVD moved recname adding to save
        # if os.path.isdir(directory):
        #    directory = os.path.join(directory, self.name)
        if os.path.exists(directory):
            m = "File named {} exists; unable to create dir".format(directory)
            raise ValueError(m)
        else:
            os.mkdir(directory)
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        for i, v in enumerate(self.signal_views):
            for k, s in v.items():
                if i == 0:
                    print(i, k)
                    s.save(directory, fmt=fmt)
                else:
                    if s is not self.signal_views[i - 1][k]:
                        print(i, k)
                        s.save(directory, prefix=f"{i:02d}.", fmt=fmt)

        # Save meta dictionary to json file. Works?
        metafilepath = directory + os.sep + self.name + ".meta.json"
        md_fh = open(metafilepath, "w")
        self._save_metadata(md_fh)

        return directory

    def save_targz(self, uri, fmt=None):
        """
        Saves all the signals (CSV/JSON pairs) in this recording
        as a .tgz file at a local URI.
        """
        directory = os.path.dirname(uri)
        if not os.path.isdir(directory):
            os.makedirs(directory, mode=0o0777)
        os.umask(0o0000)
        with open(uri, "wb") as archive:
            tgz = self.as_targz(fmt=fmt)
            archive.write(tgz.read())
            tgz.close()
        return uri

    def as_targz(self, fmt=None):
        """
        Returns a BytesIO containing all the rec's signals as a .tgz stream.
        You may either send this over HTTP or save it to a file. No temporary
        files are created in the creation of this .tgz stream.

        Example of saving an in-memory recording to disk:
            rec = Recording(...)
            with open('/some/path/test.tgz', 'wb') as fh:
                tgz = rec.as_targz()
                fh.write(tgz.read())
                tgz.close()  # Don't forget to close it!
        """
        f = io.BytesIO()  # Create a buffer
        tar = tarfile.open(fileobj=f, mode="w:gz")
        # tar = tarfile.open('/home/ivar/poopy.tgz', mode='w:gz')
        # With the tar buffer open, write meta data, then all signal files

        # save meta
        metafilebase = self.name + ".meta.json"
        md_fh = io.StringIO()
        self._save_metadata(md_fh)
        stream = io.BytesIO(md_fh.getvalue().encode())
        info = tarfile.TarInfo(os.path.join(self.name, metafilebase))
        info.uname = "nems"  # User name
        info.gname = "users"  # Group name
        info.mtime = time.time()
        info.size = stream.getbuffer().nbytes
        tar.addfile(info, stream)

        for i, v in enumerate(self.signal_views):
            for k, s in v.items():
                s.name = k
                _save_this = False
                if i == 0:
                    log.info(f"view {i}, signal {k}")
                    _save_this = True
                    prefix = ""
                else:
                    if s is not self.signal_views[i - 1][k]:
                        print(i, k)
                        prefix = f"{i:02d}."
                        _save_this = True
                if _save_this:
                    d = s.as_file_streams(fmt=fmt)  # Dict mapping filenames to streams
                    for filename, stringstream in d.items():
                        if type(stringstream) is io.BytesIO:
                            stream = stringstream
                        else:
                            stream = io.BytesIO(stringstream.getvalue().encode())
                        info = tarfile.TarInfo(
                            os.path.join(self.name, prefix + filename)
                        )
                        info.uname = "nems"  # User name
                        info.gname = "users"  # Group name
                        info.mtime = time.time()
                        info.size = stream.getbuffer().nbytes
                        tar.addfile(info, stream)

        tar.close()
        f.seek(0)
        return f

    def save_url(self, uri, compressed=False):
        """
        Saves this recording to a URL. Returns the URI if it succeeded,
        else None. Check the return code to see if the URL save failed
        and you need to save locally instead. e.g.

        # Example: Try to save remotely, or save locally if it fails
        if not rec.save_url(url):
             rec.save('/tmp/')   # Save to /tmp as a fallback
        """
        r = requests.put(uri, data=self.as_targz())
        if r.status_code == 200:
            return uri
        else:
            m = "HTTP PUT failed (Code: {}) for {}.".format(r.status_code, uri)
            log.warn(m)
            return None

    def _save_metadata(self, md_fh, fmt="%.18e"):
        """
        Save this signal to a CSV file + JSON sidecar. If desired,
        you may use optional parameter fmt (for example, fmt='%1.3e')
        to alter the precision of the floating point matrices.
        """
        json.dump(self.meta, md_fh, cls=NumpyEncoder)

    def get_signal(self, signal_name):
        """
        Returns the signal object with the given signal_name, or None
        if it was was found.

        signal_name should be a string
        """
        if signal_name in self.signals:
            return self.signals[signal_name]
        else:
            return None

    def add_signal(self, signal):
        """
        Adds the signal equal to this recording. Any existing signal
        with the same name will be overwritten. No return value.
        """
        if not isinstance(signal, SignalBase):
            import pdb

            pdb.set_trace()
            raise TypeError(
                "Recording signals must be instances of"
                " a Signal class. signal {} was type: {}".format(
                    signal.name, type(signal)
                )
            )
        self.signals[signal.name] = signal

    def _split_helper(self, fn):
        """
        For internal use only by the split_* functions.
        """
        est = {}
        val = {}
        for s in self.signals.values():
            (e, v) = fn(s)
            est[e.name] = e
            val[v.name] = v

        est = Recording(signals=est)
        val = Recording(signals=val)

        est = est.and_mask(np.isfinite(est["resp"].as_continuous()[0, :]))
        val = val.and_mask(np.isfinite(val["resp"].as_continuous()[0, :]))
        #        if 'mask' in est.signals.keys():
        #            log.info('mask exists, Merging (AND) with masks for partitioned est,val signals')
        #            m = est['mask'].as_continuous().squeeze()
        #            est = est.create_mask(np.logical_and(m,np.isfinite(est['resp'].as_continuous()[0,:])))
        #            val = val.create_mask(np.logical_and(m,np.isfinite(val['resp'].as_continuous()[0,:])))
        #        else:
        #            log.info('creating masks for partitioned est,val signals')
        #            est = est.create_mask(np.isfinite(est['resp'].as_continuous()[0,:]))
        #            val = val.create_mask(np.isfinite(val['resp'].as_continuous()[0,:]))

        return (est, val)

    def split_at_time(self, fraction):
        """
        Calls .split_at_time() on all signal objects in this recording.
        For example, fraction = 0.8 will result in two recordings,
        with 80% of the data in the left, and 20% of the data in
        the right signal. Useful for making est/val data splits, or
        truncating the beginning or end of a data set.

        FOR silly reasons having to do with the ordering of val stimuli,
          "r" is actually the beginning of the signal -- used for val
          "l" is the end, used for est
        """
        est = {}
        val = {}
        for s in self.signals.values():
            v, e = s.split_at_time(fraction)
            est[e.name] = e
            val[v.name] = v

        est = Recording(signals=est)
        val = Recording(signals=val)

        return est, val

    def split_by_epochs(self, epochs_for_est, epochs_for_val):
        """
        Returns a tuple of estimation and validation data splits: (est, val).
        Arguments should be lists of epochs that define the estimation and
        validation sets. Both est and val will have non-matching data NaN'd out.
        """
        return self._split_helper(
            lambda s: s.split_by_epochs(epochs_for_est, epochs_for_val)
        )

    def select_times(self, times, padding=0, reset_epochs=False):

        if padding != 0:
            raise NotImplementedError  # TODO

        k = list(self.signals.keys())
        newsigs = {n: s.select_times(times) for n, s in self.signals.items()}

        if reset_epochs:
            newsigs = {n: s.reset_segmented_epochs() for n, s in newsigs.items()}
            del newsigs["mask"]
        return Recording(newsigs, meta=self.meta)

    def nan_times(self, times, padding=0):

        if padding != 0:
            raise NotImplementedError  # TODO

        k = list(self.signals.keys())
        newsigs = {n: s.nan_times(times) for n, s in self.signals.items()}

        return Recording(newsigs)

    def create_mask(self, epoch=None, base_signal=None, mask_name="mask"):
        """
        inputs:
            epoch: {None, boolean, ndarray, string, list}
             if None, defaults to False
             if False, initialize mask signal to False for all times
             if True, initialize mask signal to False for all times
             if Tx1 ndarray, True where ndarray is true, False elsewhere
             if Nx2 ndarray, True in N epoch times
             if string (eoch name), mask is True for epochs with .name==string
             if list of strings (epoch names), mask is OR combo of all strings
             if list of tuples (epoch times), mask is OR combo of all epoch times

        TODO: add epochs, base signal parameters
        """

        rec = self.copy()
        if base_signal is None:
            sig_name = list(rec.signals.keys())[0]
            base_signal = rec[sig_name]

        mask = base_signal.generate_epoch_mask(epoch)

        try:
            mask_sig = base_signal._modified_copy(mask, dtype="bool")
        except AttributeError:
            # Only rasterized signals support _modified_copy
            mask_sig = base_signal.rasterize()._modified_copy(mask)
        mask_sig.name = mask_name

        rec.add_signal(mask_sig)

        return rec

    def and_mask(self, epoch, invert=False):
        """
        Make rec['mask'] == True for all epochs where current mask is also true.
        Mask is created if it doesn't exist
        See create_mask for input formats for 'epoch'

        example use:
            newrec = rec.or_mask(['ACTIVE_EXPERIMENT'])
            newrec = rec.and_mask(['REFERENCE', 'TARGET'])

            newrec['mask'] == True only during REFERENCE and TARGET epochs
            contained within ACTIVE_EXPERIMENT epochs
        """
        if "mask" not in self.signals.keys():
            rec = self.create_mask(True)
        else:
            rec = self.copy()
        and_mask = rec["mask"].generate_epoch_mask(epoch)

        # Invert
        if invert:
            and_mask = ~and_mask

        # apply and_mask to existing mask
        m = rec["mask"].as_continuous().astype("bool")
        rec["mask"] = rec["mask"]._modified_copy(m & and_mask)

        return rec

    def nan_mask(self, remove_epochs=True):
        """
        Nan-out data based on boolean signal called mask. Returns new recording
        with only data specified mask. To make mask, see "create_mask"
        :param remove_epochs: (True) if true, delete epochs that are all nan
        :return: rec : copy of self with masked periods set to nan
        """
        if "mask" not in self.signals.keys():
            warnings.warn("No mask specified, nan_mask() simply copying recording.")
            return self.copy()

        rec = self.copy()
        m = rec["mask"].copy()

        if np.sum(m._data == False) == 0:
            # mask is all true, passthrough
            return rec

        for k, sig in rec.signals.items():
            if k != "mask":
                rec[k] = sig.rasterize().nan_mask(m, remove_epochs=remove_epochs)
            else:
                rec[k] = sig.remove_epochs(m)

        return rec


## I/O functions
def load_recording_from_targz(targz):
    if os.path.exists(targz):
        with open(targz, "rb") as stream:
            return load_recording_from_targz_stream(stream)
    else:
        m = "Not a .tgz file: {}".format(targz)
        raise ValueError(m)


def load_recording_from_targz_stream(tgz_stream):
    """
    Loads the recording object from the given .tgz stream, which
    is expected to be a io.BytesIO object.
    For hdf5 files, copy to temporary directory and load with hdf5 utility
    """
    tpath = None
    meta = {}
    streams = [{}]  # For holding file streams as we unpack
    with tarfile.open(fileobj=tgz_stream, mode="r:gz") as t:
        for member in t.getmembers():
            if member.size == 0:  # Skip empty files
                continue
            basename = os.path.basename(member.name)

            # Now put it in a subdict so we can find it again
            _pieces = basename.split(".")[:-1]
            if _pieces[-1] == "epoch":
                _pieces = _pieces[:-1]
            if len(_pieces) == 2:
                v = 0
                signame = str(_pieces)
            else:
                v = int(_pieces[0])
                signame = str(_pieces[1:])
            # signame = str(basename.split('.')[0:2])

            if basename.endswith("meta.json"):
                f = io.StringIO(t.extractfile(member).read().decode("utf-8"))
                meta = json.load(f, object_hook=json_numpy_obj_hook)
                f = None
            elif basename.endswith("epoch.csv"):
                keyname = "epoch_stream"
                f = io.StringIO(t.extractfile(member).read().decode("utf-8"))

            elif basename.endswith(".csv"):
                keyname = "data_stream"
                f = io.StringIO(t.extractfile(member).read().decode("utf-8"))

            elif basename.endswith(".h5"):
                keyname = "data_stream"
                # f_in = io.BytesIO(t.extractfile(member).read())

                # current non-optimal solution. extract hdf5 file to disk and then load
                if not tpath:
                    tpath = tempfile.mktemp()
                t.extract(member, tpath)
                f = tpath + "/" + member.name

            elif basename.endswith(".json"):
                keyname = "json_stream"
                f = io.StringIO(t.extractfile(member).read().decode("utf-8"))

            else:
                m = "Unexpected file found in tgz: {} (size={})".format(
                    member.name, member.size
                )
                raise ValueError(m)

            if f is not None:
                # Ensure that we can doubly nest the streams dict
                if len(streams) < (v + 1):
                    streams.append({})
                if signame not in streams[v]:
                    streams[v][signame] = {}
                # Read out a stringIO object for each file now while it's open
                # f = io.StringIO(t.extractfile(member).read().decode('utf-8'))
                streams[v][signame][keyname] = f

    # Now that the streams are organized, convert them into signals
    # log.debug({k: streams[k].keys() for k in streams})
    signal_views = []
    previous_dict = {}
    for stream in streams:
        _sigs = [load_signal_from_streams(**sg) for sg in stream.values()]
        signals_dict = {s.name: s for s in _sigs}

        signal_views.append(previous_dict.copy())
        signal_views[-1].update(signals_dict)

        previous_dict = signal_views[-1]

    rec = Recording(signals={}, meta=meta, signal_views=signal_views)

    if tpath:
        shutil.rmtree(tpath)  # clean up if tpath is not None

    return rec


def load_recording(uri):
    """
    Loads from a local .tgz file, a local directory, from s3,
    or from an HTTP URL containing a .tgz file. Examples:

    # Load all signals in the gus016c-a2 directory
    rec = Recording.load('/home/myuser/gus016c-a2')
    rec = Recording.load('file:///home/myuser/gus016c-a2')

    # Load the local tar gz directory.
    rec = Recording.load('file:///home/myuser/gus016c-a2.tgz')

    # Load a tgz file served from a flat filesystem
    rec = Recording.load('http://potoroo/recordings/gus016c-a2.tgz')

    # Load a tgz file created by the nems-baphy interafce
    rec = Recording.load('http://potoroo/baphy/271/gus016c-a2')

    # Load from S3:
    rec = Recording.load('s3://nems0.lbhb... TODO')
    """
    if type(uri) in [PosixPath, WindowsPath]:
        uri = str(uri)
    # uri = adjust_uri_prefix(uri)

    if local_uri(uri):
        if targz_uri(uri):
            rec = load_recording_from_targz(local_uri(uri))
        else:
            rec = load_recording_from_dir(local_uri(uri))
    elif http_uri(uri):
        log.info(f"URL: {uri}")
        rec = load_recording_from_url(http_uri(uri))
    elif uri[0:6] == "s3://":
        raise NotImplementedError
    else:
        raise ValueError("Invalid URI: {}".format(uri))
    rec.uri = uri

    return rec


def load_recording_from_dir(directory_or_targz):
    """
    Loads all the signals (CSV/JSON pairs) found in DIRECTORY or
    .tgz file, and returns a Recording object containing all of them.
    """
    if os.path.isdir(directory_or_targz):
        files = list_signals(directory_or_targz)
        basepaths = [os.path.join(directory_or_targz, f) for f in files]
        signals = [load_signal(f) for f in basepaths]
        signals_dict = {s.name: s for s in signals}

        # find
        metafilepath = [
            p for p in os.listdir(directory_or_targz) if ".meta.json" in p
        ]  # signleton list
        metafilepath = os.path.join(directory_or_targz, metafilepath[0])
        with open(metafilepath, "r") as f:
            meta = json.load(f)

        return Recording(signals=signals_dict, meta=meta)
    else:
        m = "Not a directory: {}".format(directory_or_targz)
        raise ValueError(m)


def load_recording_from_url(url):
    """
    Loads the recording object from a URL. File must be tgz format.
    """
    r = requests.get(url, stream=True)
    if r.status_code == 400:
        m = "Not found on server: {}".format(url)
        raise Exception(m)
    elif not (
            r.status_code == 200
            and (
                    r.headers["content-type"] == "application/gzip"
                    or r.headers["content-type"] == "text/plain"
                    or r.headers["content-type"] == "application/x-gzip"
                    or r.headers["content-type"] == "application/x-compressed"
                    or r.headers["content-type"] == "application/x-compressed-tar"
                    or r.headers["content-type"] == "application/x-tar"
                    or r.headers["content-type"] == "application/x-tgz"
            )
    ):
        log.info("got response: {}, {}".format(r.headers, r.status_code))
        log.info(
            "status_code: %d, content-type: %s",
            r.status_code,
            r.headers["content-type"],
        )
        m = "Error loading URL: {}".format(url)
        log.error(m)
        raise Exception(m)
    obj = io.BytesIO(r.raw.read())  # Not sure why I need this!
    return load_recording_from_targz_stream(obj)


def load_recording_from_arrays(arrays, rec_name, fs, sig_names=None, signal_kwargs={}):
    """
    Generates a recording object, and the signal objects it contains,
    from a list of array-like structures of the form channels x time
    (see signal.py for more details about how arrays are represented
     by signals).

    If any of the arrays are more than 2-dimensional,
    an error will be thrown. Also pay close attention to any
    RuntimeWarnings from the signal class regarding improperly-shaped
    arrays, which may indicate that an array was passed as
    time x channels instead of the reverse.

    Arguments:
    ----------
    arrays : list of array-like
        The data to be converted to a recording of signal objects.
        Each item should be 2-dimensional and convertible to a
        numpy ndarray via np.array(x). No constraints are enforced
        on the dtype of the arrays, but in general float values
        are expected by most native NEMS functions.

    rec_name : str
        The name to be given to the new recording object. This will
        also be assigned as the recording attribute of each new signal.

    fs : int or list of ints
        The frequency of sampling of the data arrays - used to
        interconvert between real time and time bins (see signal.py).
        If int, the same fs will be assigned to each signal.
        If list, the length must match the length of arrays.

    sig_names : list of strings (optional)
        Name to attach to the signal created from
        each array. The length of this list should match that of
        arrays.
        If not specified, the signals will be given the generic
        names: ['signal1', 'signal2', ...].

    signal_kwargs : list of dicts
        Keyword arguments to be passed through to
        each signal object. The length of this list should
        match the length of arrays, and may be padded with empty
        dictionaries to ensure this constraint.
        For example:
            [{'chans': ['1 kHz', '3 kHz']}, {'chans': ['one', 'two']}, {}]
        Would assign channel names '1 kHz' and '3 kHz' to the signal
        for the first array, 'one' and 'two' for the second array,
        and no channel names (or any other arguments) for the third array.

        Valid keyword arguments are: chans, epochs, meta,
                                     and safety_checks

    Returns:
    --------
    rec : recording object
        New recording containing a signal for each array.
    """
    # Assemble and validate lists for signal construction
    arrays = [np.array(a) for a in arrays]
    for i, a in enumerate(arrays):
        if len(a.shape) != 2:
            raise ValueError(
                "Arrays should have shape chans x time."
                "Array {} had shape: {}".format(i, a.shape)
            )
    n = len(arrays)
    recs = [rec_name] * len(arrays)
    if sig_names:
        if not len(sig_names) == n:
            raise ValueError(
                "Length of sig_names must match"
                "the length of arrays.\n"
                "Got sig_names: {} and arrays: {}".format(len(sig_names), n)
            )
    else:
        sig_names = ["sig%s" % i for i in range(n)]
    if isinstance(fs, int):
        fs = [fs] * n
    else:
        if not len(fs) == n:
            raise ValueError(
                "Length of fs must match"
                "the length of arrays.\n"
                "Got fs: {} and arrays: {}".format(len(fs), n)
            )
    if not signal_kwargs:
        signal_kwargs = [{}] * n
    else:
        if not len(signal_kwargs) == n:
            raise ValueError(
                "Length of signal_kwargs must match"
                "the length of arrays.\n"
                "Got signal_kwargs: {} and arrays: {}".format(len(signal_kwargs), n)
            )

    # Construct the signals
    to_sigs = zip(fs, arrays, sig_names, recs, signal_kwargs)
    signals = [
        RasterizedSignal(fs, a, name, rec, **kw) for fs, a, name, rec, kw in to_sigs
    ]
    signals = {s.name: s for s in signals}
    # Combine into recording and return
    return Recording(signals)
