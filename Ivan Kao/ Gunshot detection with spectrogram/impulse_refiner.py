"""Frozen CNN plus a learned correction using sub-frame impulse features."""
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
import gunshot as g


class BoundedGradientBoostingClassifier(GradientBoostingClassifier):
    """Bound new Newton leaf updates; saturated priors can have tiny Hessians.

    Existing fitted trees are preserved when a classifier is resumed. The
    per-stage binary logit change cannot exceed learning_rate * max_leaf_update.
    """
    def _fit_stage(self,i,X,y,raw_predictions,sample_weight,sample_mask,random_state,X_csc=None,X_csr=None):
        if raw_predictions.shape[1]!=1:raise ValueError('Bounded continuation supports binary classification only.')
        before=raw_predictions.copy()
        result=super()._fit_stage(i,X,y,raw_predictions,sample_weight,sample_mask,random_state,X_csc=X_csc,X_csr=X_csr)
        bound=float(getattr(self,'max_leaf_update',3.))
        if not np.isfinite(bound) or bound<=0:raise ValueError('Invalid Newton update bound.')
        tree=self.estimators_[i,0]
        np.clip(tree.tree_.value,-bound,bound,out=tree.tree_.value)
        result[:]=before
        result[:,0]+=self.learning_rate*tree.predict(X,check_input=False)
        return result


class CNNPrior(ClassifierMixin,BaseEstimator):
    def fit(self,x,y,sample_weight=None):
        self.classes_=np.array([0,1]);self.n_features_in_=x.shape[1]
        return self
    def predict_proba(self,x):
        p=np.clip(x[:,0],1e-5,1-1e-5)
        return np.column_stack([1-p,p])


class RefinerPrior(ClassifierMixin,BaseEstimator):
    """Frozen earlier classifier with finite confidence for further learning."""
    def __init__(self,expert,probability_floor=.001,feature_count=None):
        self.expert=expert;self.probability_floor=probability_floor;self.feature_count=feature_count
    def fit(self,x,y,sample_weight=None):
        self.classes_=np.array([0,1]);self.n_features_in_=x.shape[1]
        return self
    def __setstate__(self,state):
        super().__setstate__(state)
        self.__dict__.setdefault('feature_count',None)
    def predict_proba(self,x):
        count=getattr(self,'feature_count',None)
        values=x if count is None else x[:,:count]
        p=np.clip(self.expert.predict_proba(values)[:,1],self.probability_floor,1-self.probability_floor)
        return np.column_stack([1-p,p])


class ExpertFeatureHead(ClassifierMixin,BaseEstimator):
    """Frozen expert probability as a feature for a separately learned head."""
    def __init__(self,expert,head):
        self.expert=expert;self.head=head
    def inputs(self,x):
        prior=self.expert.predict_proba(x[:,:self.expert.n_features_in_])[:,1]
        return np.column_stack([x,prior])
    def fit(self,x,y,sample_weight=None):
        self.head.fit(self.inputs(x),y,sample_weight=sample_weight)
        self.classes_=self.head.classes_;self.n_features_in_=x.shape[1]
        return self
    def predict_proba(self,x):return self.head.predict_proba(self.inputs(x))
    @classmethod
    def from_fitted(cls,expert,head):
        result=cls(expert,head);result.classes_=head.classes_;result.n_features_in_=head.n_features_in_-1
        return result


class ProbabilityBlend(ClassifierMixin,BaseEstimator):
    """Mix a frozen existing system with a learned correction system."""
    def __init__(self,expert,correction,weight):
        self.expert=expert;self.correction=correction;self.weight=weight
        self.classes_=np.array([0,1]);self.n_features_in_=max(expert.n_features_in_,correction.n_features_in_)
    def predict_proba(self,x):
        old=self.expert.predict_proba(x[:,:self.expert.n_features_in_])
        new=self.correction.predict_proba(x[:,:self.correction.n_features_in_])
        return (1-self.weight)*old+self.weight*new


def spectrogram_summary(spectra):
    """Peak-aligned 16-band x 10-time-bin mean/max summaries (320 values)."""
    s=np.asarray(spectra,dtype=np.float32)
    if s.ndim==4:s=s[:,0]
    if s.ndim==2:s=s[None]
    if s.ndim!=3:raise ValueError('Expected window/frequency/time spectrograms.')
    low=s.min(axis=(1,2),keepdims=True);scale=s.max(axis=(1,2),keepdims=True)-low
    s=(s-low)/np.maximum(scale,1e-8)
    w,t=np.unravel_index(int(np.argmax(s.mean(axis=1))),s.shape[::2])
    frequencies=s.shape[1];padded=((frequencies+15)//16)*16
    patch=np.zeros((padded,40),dtype=np.float32);start=t-8;left=max(0,start);right=min(s.shape[2],start+40)
    patch[:frequencies,left-start:right-start]=s[w,:,left:right]
    bins=patch.reshape(16,padded//16,10,4)
    return np.r_[bins.mean(axis=(1,3)).ravel(),bins.max(axis=(1,3)).ravel()].astype(np.float32)


def waveform_features(audio):
    """Peak-centered, amplitude-normalized features at 24 kHz.

    No filename, sample rate, source label, or original file length is a feature.
    """
    audio=np.asarray(audio,dtype=np.float64)
    peak=int(np.abs(audio).argmax());start=peak-g.SR
    x=np.zeros(g.SAMPLES,dtype=np.float64)
    left=max(0,start);right=min(len(audio),start+g.SAMPLES)
    x[left-start:right-start]=audio[left:right]
    x/=max(float(np.abs(x).max()),1e-12)
    center=g.SR;energy=x*x;total=float(energy.sum())+1e-12
    out=[]
    for milliseconds in [1,2,5,10,20,50,100,250,500,1000]:
        n=max(1,round(milliseconds*g.SR/1000))
        out.extend([float(energy[center-n:center+n].sum()/total),
                    float(energy[center:center+n].sum()/total),
                    float(energy[center-n:center].sum()/total)])
    for duration in [.02,.08,.25,1.]:
        n=round(duration*g.SR);segment=x[center-n:center+n]
        rms=np.sqrt(np.mean(segment*segment)+1e-12)
        out.extend([float(np.mean(np.abs(segment))),float(rms),
                    float(np.mean(segment**4)/(rms**4+1e-12)),
                    float(np.mean(np.abs(np.diff(segment)))),
                    float(np.mean(segment[:-1]*segment[1:]<0))])
        power=np.abs(np.fft.rfft(segment*np.hanning(len(segment)),n=16384))**2
        frequencies=np.fft.rfftfreq(16384,1/g.SR);denominator=float(power.sum())+1e-12
        bands=[0,250,500,1000,2000,4000,6000,9000,12001]
        out.extend(float(power[(frequencies>=a)&(frequencies<b)].sum()/denominator) for a,b in zip(bands[:-1],bands[1:]))
        out.extend([float((frequencies*power).sum()/denominator/g.SR),
                    float(np.exp(np.mean(np.log(power+1e-12)))/(np.mean(power)+1e-12))])
    values=np.nan_to_num(np.asarray(out,dtype=np.float32),nan=0.,posinf=1e6,neginf=-1e6)
    return values


def cnn_embedding(net,spectra):
    """Retain the frozen compact CNN's 64 learned pattern responses per recording."""
    if not isinstance(net,torch.nn.Sequential) or not isinstance(net[-1],torch.nn.Linear):
        raise ValueError('Embedding extraction requires the compact sequential CNN.')
    if net.training:raise ValueError('Embedding extraction requires evaluation mode.')
    x=torch.as_tensor(np.asarray(spectra,dtype=np.float32))
    if x.ndim!=4 or not len(x):raise ValueError('Expected nonempty window/channel/frequency/time spectra.')
    features=[]
    with torch.inference_mode():
        trunk=net[:-1]
        for batch in x.split(32):features.append(trunk(batch))
    result=torch.cat(features).amax(dim=0).numpy().astype(np.float32)
    if result.shape!=(net[-1].in_features,) or not np.isfinite(result).all():raise ValueError('Invalid CNN embedding.')
    return result


from functools import lru_cache


@lru_cache(maxsize=3)
def load_adapted_network(path,expected_sha256):
    from continual_cycles import fingerprint
    from pathlib import Path
    if fingerprint(Path(path))!=expected_sha256:raise ValueError('Adapted CNN checksum mismatch')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    net=g.model_for(checkpoint['config']);net.load_state_dict(checkpoint['state_dict']);net.eval()
    return checkpoint,net


def adapted_cnn_features(config,spectra):
    checkpoint,net=load_adapted_network(config['adapted_cnn_checkpoint'],config['adapted_cnn_sha256'])
    x=torch.as_tensor(np.asarray(spectra,dtype=np.float32))
    with torch.inference_mode():probability=float(g.recording_logits(net,[x],torch.device('cpu')).sigmoid().item())
    return np.r_[probability,cnn_embedding(net,spectra)].astype(np.float32)
