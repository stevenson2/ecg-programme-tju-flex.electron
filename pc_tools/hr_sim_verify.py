"""
ECG心率检测验证仿真器
=====================
复现ESP32上的完整信号处理链路，验证心率检测精度

用法: python pc_tools/hr_sim_verify.py
"""

import numpy as np
from scipy import signal
import math

FS = 250.0
TS = 1.0 / FS

# ============ 1. ECG信号生成 ============
class ECGFrameGenerator:
    def __init__(self):
        self.cycle_length = 200
        self.idx = 0
        self.rand_seed = 42
        self.clean_val = 0.0
        self.DC_OFFSET = 1.65
        self.PL_50HZ_AMP = 0.145
        self.PL_100HZ_AMP = 0.040
        self.BL_RESP_AMP = 0.060; self.BL_SLOW_AMP = 0.035; self.BL_VSLOW_AMP = 0.025
        self.EMG_SCALE = 0.060; self.MOTION_AMP = 0.065; self.SPIKE_AMP = 0.150; self.SYS_NOISE_SCALE = 0.035
        self.motion_decay = 0.0; self.motion_countdown = 0; self.emg_envelope = 0.0
        self.params = {'P':(0.25,0.18,0.030),'Q':(-0.10,0.30,0.020),'R':(1.20,0.33,0.015),
                       'S':(-0.15,0.37,0.025),'T':(0.30,0.55,0.060)}
    
    def _gauss(self, x, amp, center, sigma):
        t = (x - center) / sigma
        return amp * np.exp(-0.5 * t * t)
    
    def _rand(self):
        self.rand_seed = (self.rand_seed * 1103515245 + 12345) & 0x7FFFFFFF
        return self.rand_seed / 2147483648.0
    
    def generate(self):
        t = self.idx / self.cycle_length
        self.clean_val = sum(self._gauss(t, *p) for p in self.params.values())
        a50 = 2*math.pi*50*self.idx/FS
        a100 = 2*math.pi*100*self.idx/FS
        pl = self.PL_50HZ_AMP*math.sin(a50)+self.PL_100HZ_AMP*math.sin(a100+0.5)
        aR = 2*math.pi*0.25*self.idx/FS; aS = 2*math.pi*0.10*self.idx/FS; aV = 2*math.pi*0.03*self.idx/FS
        bl = self.BL_RESP_AMP*math.sin(aR)+self.BL_SLOW_AMP*math.sin(aS+1.3)+self.BL_VSLOW_AMP*math.sin(aV+2.7)
        self.emg_envelope += (self._rand()-0.5)*0.1
        self.emg_envelope = max(0, min(1, self.emg_envelope))
        emg = sum(self._rand() for _ in range(12))
        emg = (emg-6.0)*(self.EMG_SCALE+(0.100 if self._rand()<0.02 else 0))*self.emg_envelope
        motion = 0.0
        if self.motion_countdown <= 0 and self._rand() < 0.002:
            self.motion_decay = (self._rand()-0.5)*2.0*self.MOTION_AMP
            self.motion_countdown = 50+int(self._rand()*200)
        if self.motion_countdown > 0:
            motion = self.motion_decay; self.motion_decay *= 0.970; self.motion_countdown -= 1
        elec = (self._rand()-0.5)*2.0*self.SPIKE_AMP if self._rand()<0.005 else 0.0
        sys_noise = sum(self._rand() for _ in range(12))
        sys_noise = (sys_noise-6.0)*self.SYS_NOISE_SCALE
        noisy = self.clean_val+pl+bl+emg+motion+elec+sys_noise+self.DC_OFFSET
        self.idx = (self.idx+1)%self.cycle_length
        return noisy, self.clean_val

# ============ 2. 滤波器 ============
class FilterChain:
    def __init__(self):
        self.hp_b,self.hp_a = signal.butter(2,0.5,'high',fs=FS,output='ba')
        self.lp_b,self.lp_a = signal.butter(2,40,'low',fs=FS,output='ba')
        w0=2*math.pi*50/FS; c50=math.cos(w0); s50=math.sin(w0)
        a20=s50/40; self.n20_b=np.r_[1,-2*c50,1]/(1+a20); self.n20_a=np.r_[1,-2*c50/(1+a20),(1-a20)/(1+a20)]
        a30=s50/60; self.n30_b=np.r_[1,-2*c50,1]/(1+a30); self.n30_a=np.r_[1,-2*c50/(1+a30),(1-a30)/(1+a30)]
        w0100=2*math.pi*100/FS; a15=math.sin(w0100)/30; c100=math.cos(w0100)
        self.n100_b=np.r_[1,-2*c100,1]/(1+a15); self.n100_a=np.r_[1,-2*c100/(1+a15),(1-a15)/(1+a15)]
        self.zi_hp=signal.lfilter_zi(self.hp_b,self.hp_a)*0; self.zi_lp=signal.lfilter_zi(self.lp_b,self.lp_a)*0
        self.zi_n20=signal.lfilter_zi(self.n20_b,self.n20_a)*0; self.zi_n30=signal.lfilter_zi(self.n30_b,self.n30_a)*0
        self.zi_n100=signal.lfilter_zi(self.n100_b,self.n100_a)*0
    def filter(self,x):
        y,self.zi_hp=signal.lfilter(self.hp_b,self.hp_a,[x],zi=self.zi_hp)
        y,self.zi_lp=signal.lfilter(self.lp_b,self.lp_a,y,zi=self.zi_lp)
        y,self.zi_n20=signal.lfilter(self.n20_b,self.n20_a,y,zi=self.zi_n20)
        y,self.zi_n30=signal.lfilter(self.n30_b,self.n30_a,y,zi=self.zi_n30)
        y,self.zi_n100=signal.lfilter(self.n100_b,self.n100_a,y,zi=self.zi_n100)
        return y[0]

# ============ 3. 心率检测 ============
class HRDetector:
    def __init__(self):
        self.bpf_lp_b,self.bpf_lp_a = signal.butter(2,15,'low',fs=FS,output='ba')
        self.bpf_hp_b,self.bpf_hp_a = signal.butter(2,5,'high',fs=FS,output='ba')
        self.zi_bpf_lp=signal.lfilter_zi(self.bpf_lp_b,self.bpf_lp_a)*0
        self.zi_bpf_hp=signal.lfilter_zi(self.bpf_hp_b,self.bpf_hp_a)*0
        self.prev=0.0; self.mwi_buf=np.zeros(38); self.mwi_idx=0; self.mwi_sum=0.0
        self.mwi_p=0.0; self.mwi_p2=0.0
        self.thr=0.002; self.sp=0.002; self.np=0.0006
        self.rr_buf=[]; self.ssb=0; self.bc=0; self.bpm_h=[]
    def qrs_bpf(self,x):
        y,self.zi_bpf_lp=signal.lfilter(self.bpf_lp_b,self.bpf_lp_a,[x],zi=self.zi_bpf_lp)
        y,self.zi_bpf_hp=signal.lfilter(self.bpf_hp_b,self.bpf_hp_a,y,zi=self.zi_bpf_hp)
        return y[0]
    def process(self,x):
        qrs=self.qrs_bpf(x); d=qrs-self.prev; self.prev=qrs; sq=d*d
        self.mwi_sum-=self.mwi_buf[self.mwi_idx]; self.mwi_buf[self.mwi_idx]=sq; self.mwi_sum+=sq
        self.mwi_idx=(self.mwi_idx+1)%38; mwi=self.mwi_sum/38.0
        pk=(self.mwi_p>self.mwi_p2)and(self.mwi_p>mwi); bd=False; bpm=0
        if pk:
            pv=self.mwi_p; rr=self.ssb*TS
            if pv>self.thr and pv>self.np*2.0:
                self.rr_buf.append(rr)
                if len(self.rr_buf)>8:self.rr_buf.pop(0)
                self.bc+=1; self.ssb=0; bd=True
                self.sp=0.125*pv+0.875*self.sp; d2=self.sp-self.np
                if d2<0.001:d2=0.001; self.thr=self.np+0.40*d2
            else: self.np=0.125*pv+0.875*self.np
        self.ssb+=1; self.mwi_p2=self.mwi_p; self.mwi_p=mwi
        if bd and len(self.rr_buf)>=3:
            mr=np.median(self.rr_buf)
            if mr>0.3: bpm=int(60.0/mr+0.5)
            if 30<=bpm<=200:self.bpm_h.append(bpm)
        return bd,bpm

# ============ 4. 仿真 ============
print("="*60)
print("ECG心率检测验证仿真器")
print("(复现ESP32信号处理链路)")
print("="*60)

ecg=ECGFrameGenerator(); filt=FilterChain(); hr=HRDetector()

# 梳状
cb=np.zeros(5); ci=0
DUR=30
N=int(FS*DUR)

dbpm=[]; ts=[]
for i in range(N):
    noisy,clean=ecg.generate()
    nn=noisy-1.65
    cs=sum(cb)-cb[ci]; cb[ci]=nn; cs+=nn; ci=(ci+1)%5; nc=cs/5.0
    fl=filt.filter(nc)
    beat,bpm=hr.process(fl)
    if beat and bpm>0:
        dbpm.append(bpm)
        ts.append(i/FS)

exp=int(DUR*75/60)
print(f"\n采集: {N}样本 ({DUR}秒, 75BPM={exp}心拍)")
print(f"检测: {len(dbpm)}心拍")

if len(dbpm)>0:
    cov=len(dbpm)/exp*100
    print(f"覆盖: {cov:.1f}%")
    avg=np.mean(dbpm); med=np.median(dbpm); sd=np.std(dbpm)
    w3=sum(1 for b in dbpm if abs(b-75)<=3)/len(dbpm)*100
    print(f"BPM: 平均={avg:.1f}, 中位={med:.1f}, 标准差={sd:.1f}")
    print(f"精度±3BPM: {w3:.1f}%")
    if cov>=80 and w3>=85: print("\n✅ 心率检测性能良好")
    elif cov>=50: print("\n⚠️ 尚可接受")
    else: print("\n❌ 需要改进")
else:
    print("❌ 未检测到心拍!")